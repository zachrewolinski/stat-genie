from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy import stats

# Attempt to read example CSV if available; do not fail import if missing.
try:
    _sample_df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/noperturb_output/soccer.csv')
except Exception:
    _sample_df = None


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset for modeling the relationship between skin tone and red cards.

    Steps:
    - Make a working copy and drop rows with missing essential data (rater1, rater2, games, redCards, refNum).
    - Compute the average skin rating (SkinToneAvg) from rater1 and rater2.
    - Create a binary DarkSkin indicator by classifying the top range as Dark and the bottom range as Light, excluding ambiguous/middle observations.
      We use thresholds on the normalized 0-1 scale: Light if avg <= 0.4, Dark if avg >= 0.6; others labeled 'Ambiguous' and removed.
    - Parse birthday to compute age at season reference date (2013-01-01) and create an 'age' column.
    - Ensure games > 0 (required for offset). Create log_games = log(games) for model offset.
    - Keep only the columns required for the model.

    Returns the transformed dataframe ready for modeling.
    """
    df = df.copy()

    # Essential columns check / dropna
    required_cols = ['rater1', 'rater2', 'games', 'redCards', 'refNum']
    df = df.dropna(subset=required_cols)

    # Compute average skin tone rating
    df['SkinToneAvg'] = (df['rater1'].astype(float) + df['rater2'].astype(float)) / 2.0

    # Classify into Dark vs Light vs Ambiguous.
    def classify_skin(x):
        if pd.isnull(x):
            return 'Ambiguous'
        if x <= 0.4:
            return 'Light'
        elif x >= 0.6:
            return 'Dark'
        else:
            return 'Ambiguous'

    df['SkinToneClass'] = df['SkinToneAvg'].apply(classify_skin)

    # Keep only clear Dark vs Light contrasts
    df = df[df['SkinToneClass'].isin(['Dark', 'Light'])].copy()

    # Binary indicator for Dark (1) vs Light (0)
    df['DarkSkin'] = (df['SkinToneClass'] == 'Dark').astype(int)

    # Parse birthday and compute age at reference date (season midpoint)
    # birthday format in schema: dd.mm.yyyy
    if 'birthday' in df.columns:
        df['birthday_parsed'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y', errors='coerce')
    else:
        df['birthday_parsed'] = pd.NaT
    ref_date = pd.to_datetime('2013-01-01')
    df['age'] = (ref_date - df['birthday_parsed']).dt.days / 365.25

    # Keep only rows with a valid numeric games > 0 for offset
    df['games'] = pd.to_numeric(df['games'], errors='coerce')
    df = df[df['games'] > 0].copy()

    # Create log_games column for offset in count model
    df['log_games'] = np.log(df['games'])

    # Ensure numeric controls are numeric
    for col in ['height', 'weight', 'yellowCards', 'yellowReds', 'meanIAT', 'meanExp', 'redCards']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Final selection of columns required for modeling
    required_for_model = [
        'redCards', 'games', 'log_games', 'DarkSkin', 'SkinToneAvg',
        'height', 'weight', 'age', 'yellowCards', 'yellowReds',
        'position', 'leagueCountry', 'meanIAT', 'meanExp', 'refNum'
    ]

    # Keep columns that exist in the incoming dataframe; missing controls will be left as NaN and the model will drop them
    present_cols = [c for c in required_for_model if c in df.columns]
    df = df[present_cols].copy()

    # Drop rows with missing DV or IV or essential controls (redCards, DarkSkin, log_games, refNum are essential)
    essential = ['redCards', 'DarkSkin', 'log_games', 'refNum']
    df = df.dropna(subset=essential)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# Helper wrapper to present clustered robust results in a consistent interface
class ClusteredResult:
    def __init__(self, base_result, cov, param_index):
        """
        base_result: original statsmodels results object (for access to things like aic, llf, and summary)
        cov: covariance matrix as numpy array
        param_index: index/labels for parameters
        """
        self._base = base_result
        self.cov_params = pd.DataFrame(cov, index=param_index, columns=param_index)
        self.params = pd.Series(base_result.params, index=param_index)
        self.bse = pd.Series(np.sqrt(np.diag(cov)), index=param_index)
        # t-values and p-values using normal approximation
        # Guard against division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            self.tvalues = self.params / self.bse
        self.pvalues = 2 * (1 - stats.norm.cdf(np.abs(self.tvalues.fillna(0))))
        # Expose some common attributes from base result
        for attr in ['model', 'aic', 'llf', 'deviance', 'df_resid', 'df_model']:
            if hasattr(base_result, attr):
                setattr(self, attr, getattr(base_result, attr))

    def summary(self):
        # Return base summary text with a note that cov_params are clustered
        base_summary = self._base.summary()
        return f"{base_summary}\n\nNote: Standard errors, t-values, and p-values are computed using clustered covariance provided in this wrapper."


def _aligned_cluster_groups(result, df, group_col='refNum'):
    """
    Align cluster group labels to the observations actually used in the fitted model.

    statsmodels drops rows with missing data when fitting; cov_cluster requires group labels
    corresponding exactly to the observations used in the fit. This helper extracts the row
    labels used by the model and returns the matching group labels from the original dataframe.
    """
    try:
        # row_labels is typically a list/Index of index labels of the original df that were used
        row_labels = getattr(result.model.data, 'row_labels', None)

        if row_labels is None:
            # Fallback: try other common attributes
            row_labels = getattr(result.model.data, 'orig_row_labels', None)

        if row_labels is None:
            # As a last resort, if the model exposes the index of the exog/endog, try to infer from that
            try:
                # result.model.data.row_labels may be missing in some versions; use model.data._get_endog_names?
                row_labels = result.model.data.exog.index.tolist()  # type: ignore
            except Exception:
                row_labels = None

        if row_labels is None:
            # Can't determine used rows; assume df corresponds exactly (best-effort)
            groups = df[group_col].values
        else:
            # row_labels may be list of index values (could be ints or strings)
            # Use .loc to extract matching rows; this preserves order used in the fit
            groups = df.loc[row_labels, group_col].values

        # Ensure the length matches the number of observations used in the model
        n_used = int(getattr(result.model.data, 'nrow', getattr(result.model, 'nobs', None) or getattr(result, 'nobs', None) or (result.model.endog.shape[0] if hasattr(result.model, 'endog') else None)))
        if n_used is not None and len(groups) != n_used:
            # Try boolean mask approach if direct indexing didn't match
            if row_labels is not None:
                mask = df.index.isin(row_labels)
                groups = df.loc[mask, group_col].values

        return groups
    except Exception:
        # On any failure, fallback to entire df column (may cause cov_cluster to raise)
        return df[group_col].values


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative binomial regression for count outcome redCards with exposure games (log offset).

    Model specification:
      redCards ~ DarkSkin + height + weight + age + yellowCards + yellowReds + C(position) + C(leagueCountry) + meanIAT + meanExp
    Offset: log_games (log of games in the dyad)

    We cluster standard errors at the referee level (refNum) to account for within-referee dependence.

    Returns a results-like object with clustered robust covariances.
    """
    # Define formula. C(position) and C(leagueCountry) create categorical dummies automatically.
    formula = (
        'redCards ~ DarkSkin + height + weight + age + yellowCards + yellowReds '
        '+ C(position) + C(leagueCountry) + meanIAT + meanExp'
    )

    # Fit Negative Binomial GLM with offset (log of games). Use try/except to fallback to Poisson if NB fails.
    try:
        glm_nb = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=df['log_games'])
        res_nb = glm_nb.fit()
        # Clustered covariance by referee - align groups to rows used in the fit
        groups = _aligned_cluster_groups(res_nb, df, group_col='refNum')
        clustered_cov = cov_cluster(res_nb, groups)
        param_index = res_nb.params.index
        return ClusteredResult(res_nb, clustered_cov, param_index)
    except Exception:
        glm_p = smf.glm(formula=formula, data=df, family=sm.families.Poisson(), offset=df['log_games'])
        res_p = glm_p.fit()
        groups = _aligned_cluster_groups(res_p, df, group_col='refNum')
        clustered_cov = cov_cluster(res_p, groups)
        param_index = res_p.params.index
        return ClusteredResult(res_p, clustered_cov, param_index)