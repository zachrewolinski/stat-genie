from typing import Any, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy import stats


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the original dataset into a dataframe suitable for modeling.

    Steps:
    - Rename columns from feature* to meaningful names used in the model.
    - Parse birthdate and compute Age (years) at season reference date (2013-01-01, midpoint of 2012-13 season).
    - Keep dyads with at least one rater available (drop rows where both raters are missing) because skin tone cannot be evaluated otherwise.
    - Compute SkinToneAvg and a binary DarkSkin indicator using a threshold on the averaged normalized rating.
    - Drop dyads with Matches <= 0 or missing RedCards (exposure must be positive).
    - Compute log_Matches (offset) and ensure types are correct.
    - Fill missing position values with 'Unknown'.

    Returns the transformed dataframe containing all columns used by the statistical model.
    """
    df = df.copy()

    # 1) Rename columns for clarity (these exact names are used in the modeling code and cvars)
    rename_map = {
        'feature1': 'PlayerShortName',
        'feature2': 'PlayerName',
        'feature3': 'Club',
        'feature4': 'LeagueCountry',
        'feature5': 'Birthdate',
        'feature6': 'Height_cm',
        'feature7': 'Weight_kg',
        'feature8': 'Position',
        'feature9': 'Matches',
        'feature10': 'Wins',
        'feature11': 'Ties',
        'feature12': 'Losses',
        'feature13': 'Goals',
        'feature14': 'YellowCards',
        'feature15': 'YellowRedCards',
        'feature16': 'RedCards',
        'feature17': 'PhotoID',
        'feature18': 'SkinRater1',
        'feature19': 'SkinRater2',
        'feature20': 'RefereeID',
        'feature21': 'RefCountryID',
        'feature22': 'ImplicitBias',
        'feature23': 'IAT_n',
        'feature24': 'IAT_se',
        'feature25': 'ExplicitBias',
        'feature26': 'Exp_n',
        'feature27': 'Exp_se'
    }
    df = df.rename(columns=rename_map)

    # 2) Parse dates and compute Age at season midpoint
    # Birthdate is dd.mm.yyyy according to schema
    if 'Birthdate' in df.columns:
        df['Birthdate'] = pd.to_datetime(df['Birthdate'], format='%d.%m.%Y', errors='coerce')
        season_mid = pd.to_datetime('2013-01-01')
        df['Age'] = (season_mid - df['Birthdate']).dt.days / 365.25
    else:
        df['Birthdate'] = pd.NaT
        df['Age'] = np.nan

    # 3) Keep rows where at least one rater is present (we need a skin rating)
    if 'SkinRater1' in df.columns or 'SkinRater2' in df.columns:
        df = df.dropna(subset=['SkinRater1', 'SkinRater2'], how='all').copy()
    else:
        # No raters at all -> return empty dataframe with expected columns later
        df = df.iloc[0:0].copy()

    # 4) Compute average skin rating and binary dark-skin indicator
    if 'SkinRater1' not in df.columns:
        df['SkinRater1'] = np.nan
    if 'SkinRater2' not in df.columns:
        df['SkinRater2'] = np.nan

    df['SkinToneAvg'] = df[['SkinRater1', 'SkinRater2']].mean(axis=1)

    # Threshold for 'dark' vs 'light' — set at 0.60 on the normalized 0-1 scale
    df['DarkSkin'] = (df['SkinToneAvg'] >= 0.60).astype(int)

    # 5) Clean up matches and red cards: remove dyads with zero or missing matches (exposure must be positive)
    # Ensure numeric types
    if 'Matches' in df.columns:
        df['Matches'] = pd.to_numeric(df['Matches'], errors='coerce')
    else:
        df['Matches'] = np.nan

    if 'RedCards' in df.columns:
        df['RedCards'] = pd.to_numeric(df['RedCards'], errors='coerce')
    else:
        df['RedCards'] = np.nan

    # Drop rows with missing RedCards or Matches <= 0
    df = df.dropna(subset=['RedCards', 'Matches'])
    df = df[df['Matches'] > 0].copy()

    # 6) Exposure offset: log of matches
    df['log_Matches'] = np.log(df['Matches'].astype(float))

    # 7) Fill missing controls sensibly
    if 'Goals' in df.columns:
        df['Goals'] = pd.to_numeric(df['Goals'], errors='coerce').fillna(0)
    else:
        df['Goals'] = 0

    if 'YellowCards' in df.columns:
        df['YellowCards'] = pd.to_numeric(df['YellowCards'], errors='coerce').fillna(0)
    else:
        df['YellowCards'] = 0

    if 'YellowRedCards' in df.columns:
        df['YellowRedCards'] = pd.to_numeric(df['YellowRedCards'], errors='coerce').fillna(0)
    else:
        df['YellowRedCards'] = 0

    if 'ImplicitBias' in df.columns:
        df['ImplicitBias'] = pd.to_numeric(df['ImplicitBias'], errors='coerce')
    else:
        df['ImplicitBias'] = np.nan

    if 'ExplicitBias' in df.columns:
        df['ExplicitBias'] = pd.to_numeric(df['ExplicitBias'], errors='coerce')
    else:
        df['ExplicitBias'] = np.nan

    # Position as categorical; fill missing with 'Unknown'
    if 'Position' in df.columns:
        df['Position'] = df['Position'].fillna('Unknown').astype(str)
    else:
        df['Position'] = 'Unknown'

    # RefereeID: keep as-is but coerce to numeric where possible
    if 'RefereeID' in df.columns:
        df['RefereeID'] = pd.to_numeric(df['RefereeID'], errors='coerce')

    # 8) Keep only the columns necessary for modeling (but keep identifiers useful for clustering/inspection)
    keep_cols = [
        'PlayerShortName', 'PlayerName', 'Club', 'LeagueCountry', 'Birthdate', 'Age',
        'Height_cm', 'Weight_kg', 'Position',
        'Matches', 'log_Matches', 'RedCards',
        'Goals', 'YellowCards', 'YellowRedCards',
        'SkinRater1', 'SkinRater2', 'SkinToneAvg', 'DarkSkin', 'PhotoID',
        'RefereeID', 'RefCountryID', 'ImplicitBias', 'ExplicitBias',
        'IAT_n', 'IAT_se', 'Exp_n', 'Exp_se'
    ]

    # Some of these columns may not exist if data is malformed; select intersection
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Final dtype enforcement for modeling columns
    # DarkSkin must be integer 0/1
    if 'DarkSkin' in df.columns:
        df['DarkSkin'] = df['DarkSkin'].astype(int)

    # RefereeID numeric for clustering (keep as numeric; missing values may remain and will be handled in model)
    if 'RefereeID' in df.columns:
        df['RefereeID'] = pd.to_numeric(df['RefereeID'], errors='coerce')

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fits a negative-binomial regression predicting number of red cards (dyad-level) with an offset for number of matches.

    Model specification:
    RedCards ~ DarkSkin + SkinToneAvg + ImplicitBias + ExplicitBias + DarkSkin:ImplicitBias
               + Goals + YellowCards + YellowRedCards + Age + C(Position)
    Offset: log_Matches (so the model estimates red-card rate per match)

    We compute clustered robust standard errors at the referee level to account for within-referee dependence.

    Returns: an object containing the fitted results and clustered covariance matrix (and a .summary() method).
    """
    # Work on a copy to avoid mutating input
    df_work = df.copy()

    # Ensure required columns exist (by name). They must exist in the final dataframe contract.
    required = ['RedCards', 'DarkSkin', 'SkinToneAvg', 'ImplicitBias', 'ExplicitBias',
                'Goals', 'YellowCards', 'YellowRedCards', 'Age', 'Position', 'log_Matches', 'RefereeID', 'Matches']
    missing = [c for c in required if c not in df_work.columns]
    if len(missing) > 0:
        raise ValueError('Missing required columns for modeling: ' + ', '.join(missing))

    # Drop rows with missing essential values for modeling, especially RefereeID which is required for clustering.
    df_work = df_work.dropna(subset=['RefereeID', 'RedCards', 'log_Matches'])

    # Ensure numeric types where necessary
    df_work['RedCards'] = pd.to_numeric(df_work['RedCards'], errors='coerce')
    df_work['log_Matches'] = pd.to_numeric(df_work['log_Matches'], errors='coerce')

    # Drop any rows that became NA after coercion
    df_work = df_work.dropna(subset=['RedCards', 'log_Matches', 'RefereeID'])

    # Convert RefereeID to a stable type but keep original values; we'll map to integer codes for clustering
    # Keep the index to allow alignment with the model's internal row labels
    # Formula: include categorical Position as C(Position). Interaction DarkSkin:ImplicitBias tests moderation.
    formula = (
        'RedCards ~ DarkSkin + SkinToneAvg + ImplicitBias + ExplicitBias + DarkSkin:ImplicitBias '
        '+ Goals + YellowCards + YellowRedCards + Age + C(Position)'
    )

    # Fit a Negative Binomial GLM with offset = log_Matches (exposure = Matches)
    glm_nb = smf.glm(formula=formula,
                     data=df_work,
                     family=sm.families.NegativeBinomial(),
                     offset=df_work['log_Matches'])

    results = glm_nb.fit()

    # Compute clustered robust covariance matrix by RefereeID
    # Need to ensure groups align with the rows used in the fitted model.
    # results.model.data.row_labels gives the index labels of the rows used in fitting.
    try:
        row_labels = results.model.data.row_labels
    except Exception:
        # Fallback: assume all rows of df_work were used
        row_labels = df_work.index

    # Align groups to rows used in the model
    try:
        groups_raw = df_work.loc[row_labels, 'RefereeID']
    except Exception:
        # If row_labels are positional integers that do not match index labels, try positional iloc
        try:
            row_positions = np.asarray(row_labels, dtype=int)
            groups_raw = df_work.iloc[row_positions]['RefereeID']
        except Exception:
            # As a last resort, assume df_work was used in full and take its RefereeID column
            groups_raw = df_work['RefereeID']

    # Convert groups to categorical codes (non-negative integers) for cov_cluster
    groups_cat = pd.Categorical(groups_raw)
    groups = groups_cat.codes

    # Ensure there are no missing group codes (which would be coded as -1)
    if np.any(groups < 0):
        # drop rows with missing RefereeID in the set used by the model and refit
        valid_mask = groups >= 0
        # If dropping would remove all rows, raise error
        if np.sum(valid_mask) == 0:
            raise ValueError("All rows used in the model have missing RefereeID; cannot compute clustered SEs.")
        # Filter df_work to only keep rows with valid referee IDs and refit the model
        df_work = df_work.loc[df_work.index.isin(np.asarray(row_labels)[valid_mask])]
        glm_nb = smf.glm(formula=formula,
                         data=df_work,
                         family=sm.families.NegativeBinomial(),
                         offset=df_work['log_Matches'])
        results = glm_nb.fit()
        try:
            row_labels = results.model.data.row_labels
            groups_raw = df_work.loc[row_labels, 'RefereeID']
        except Exception:
            groups_raw = df_work['RefereeID']
        groups = pd.Categorical(groups_raw).codes

    # Now groups length should match the number of observations used by the model
    n_model_obs = results.model.endog.shape[0]
    if len(groups) != n_model_obs:
        # As a safe-guard attempt to align by taking only the first n_model_obs codes if possible
        if len(groups) > n_model_obs:
            groups = groups[:n_model_obs]
        else:
            raise ValueError("Could not align referee group identifiers to model observations for clustering.")

    cov_clust = cov_cluster(results, groups)

    # Compute clustered standard errors, z-stats, p-values, and 95% CIs (normal approximation)
    params = results.params
    clustered_bse = np.sqrt(np.diag(cov_clust))
    with np.errstate(divide='ignore', invalid='ignore'):
        z_vals = params / clustered_bse
    p_vals = 2 * (1 - stats.norm.cdf(np.abs(z_vals)))
    ci_lower = params - 1.96 * clustered_bse
    ci_upper = params + 1.96 * clustered_bse

    summary_df = pd.DataFrame({
        'coef': params,
        'clustered_se': clustered_bse,
        'z': z_vals,
        'P>|z|': p_vals,
        '[0.025': ci_lower,
        '0.975]': ci_upper
    })

    # Create a wrapper object to return that exposes the original results and the clustered covariance,
    # and provides a .summary() method similar to statsmodels' results objects.
    class ClusteredResults:
        def __init__(self, base_results, cov, summary_table: pd.DataFrame):
            self.results = base_results
            self.cov_cluster = cov
            self.clustered_summary = summary_table

        def summary(self):
            # Print the original model summary
            print(self.results.summary())
            # Then print clustered SE table
            print("\nClustered (by RefereeID) standard errors and inference (normal approx):")
            print(self.clustered_summary.to_string(float_format=lambda x: f"{x:0.4f}"))

        # Expose convenient attributes
        @property
        def params(self):
            return self.results.params

        @property
        def bse(self):
            # Return clustered bse
            return np.sqrt(np.diag(self.cov_cluster))

        @property
        def pvalues(self):
            return self.clustered_summary['P>|z|']

        @property
        def cov_params(self):
            return self.cov_cluster

    clustered_results = ClusteredResults(results, cov_clust, summary_df)

    # Print brief summary for user
    clustered_results.summary()

    return clustered_results


if __name__ == '__main__':
    # Example usage when running this file as a script.
    # Note: replace the path below with a real CSV path if you want to run this.
    try:
        example_df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/anonymize_output/soccer.csv')
        transformed = transform(example_df)
        _ = model(transformed)
    except FileNotFoundError:
        pass