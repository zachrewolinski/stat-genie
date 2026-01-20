from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hc1
from scipy import stats

# Note: The dataset is expected to be read by the caller; example read was in the original file.
# The two required functions below must remain with the specified signatures.


def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure required numeric columns exist and drop rows missing the dependent or key IVs
    # We need redCards (DV), games (exposure), and the two raters (rater1, rater2)
    df = df.dropna(subset=['redCards', 'games', 'rater1', 'rater2'])

    # Convert redCards and games to numeric; coerce errors and drop any remaining NA
    df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce')
    df['games'] = pd.to_numeric(df['games'], errors='coerce')
    df = df.dropna(subset=['redCards', 'games'])

    # Drop dyads with zero games (can't define a rate / offset)
    df = df[df['games'] > 0]

    # Create continuous skin tone measure (average of two raters)
    df['rater1'] = pd.to_numeric(df['rater1'], errors='coerce')
    df['rater2'] = pd.to_numeric(df['rater2'], errors='coerce')
    df['skin_avg'] = (df['rater1'] + df['rater2']) / 2.0

    # Normalize skin_avg to range 0-1 (min-max). If constant, set to 0.5.
    if df['skin_avg'].notnull().any():
        s_min = df['skin_avg'].min()
        s_max = df['skin_avg'].max()
        if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
            df['skin_avg'] = 0.5
        else:
            df['skin_avg'] = (df['skin_avg'] - s_min) / (s_max - s_min)
    else:
        df['skin_avg'] = np.nan

    # Create binary dark vs light using a median split of skin_avg
    median_skin = df['skin_avg'].median(skipna=True)
    df['skin_dark'] = (df['skin_avg'] >= median_skin).astype(int)

    # Parse birthday into datetime and compute approximate age in 2013 (season year)
    # birthday column format is dd.mm.yyyy
    df['birthday'] = pd.to_datetime(df.get('birthday'), dayfirst=True, errors='coerce')
    df['age'] = 2013 - df['birthday'].dt.year

    # Impute height and weight with medians when missing and store as new columns
    df['height_imputed'] = pd.to_numeric(df.get('height'), errors='coerce')
    df['weight_imputed'] = pd.to_numeric(df.get('weight'), errors='coerce')
    if df['height_imputed'].isnull().any():
        med_h = df['height_imputed'].median(skipna=True)
        df['height_imputed'] = df['height_imputed'].fillna(med_h)
    if df['weight_imputed'].isnull().any():
        med_w = df['weight_imputed'].median(skipna=True)
        df['weight_imputed'] = df['weight_imputed'].fillna(med_w)

    # Ensure meanIAT and meanExp are numeric; keep rows with at least one of these (they are contextual controls)
    df['meanIAT'] = pd.to_numeric(df.get('meanIAT'), errors='coerce')
    df['meanExp'] = pd.to_numeric(df.get('meanExp'), errors='coerce')
    # We will not drop rows missing both; they are contextual controls (original code kept rows even if some NA)

    # Keep position and leagueCountry as-is (categorical). Ensure they are strings and fill NA with 'Unknown'
    df['position'] = df.get('position').fillna('Unknown').astype(str)
    df['leagueCountry'] = df.get('leagueCountry').fillna('Unknown').astype(str)

    # Keep identifiers for clustering and diagnostics
    df['refNum'] = df['refNum']
    df['playerShort'] = df['playerShort']

    # Final: keep only the columns necessary for modeling (but return full df copy for flexibility)
    # Here we keep original and derived columns used in modeling
    keep_cols = [
        'playerShort', 'refNum', 'redCards', 'games',
        'skin_avg', 'skin_dark', 'meanIAT', 'meanExp',
        'age', 'height_imputed', 'weight_imputed', 'position', 'leagueCountry'
    ]

    existing_keep = [c for c in keep_cols if c in df.columns]
    return df[existing_keep].reset_index(drop=True)


def model(df: pd.DataFrame) -> Any:
    # Build design matrix for regression using the transformed dataframe returned by transform()
    # Outcome: redCards (count). Exposure / offset: games (modeling red-card rate per game).

    # Copy to avoid modifying caller data
    data = df.copy()

    # Ensure required columns are present
    # Create categorical dummies for position and leagueCountry (drop first to avoid multicollinearity)
    pos_dummies = pd.get_dummies(data['position'].astype(str), prefix='pos', drop_first=True)
    league_dummies = pd.get_dummies(data['leagueCountry'].astype(str), prefix='league', drop_first=True)

    # Construct X with controls
    numeric_cols = ['skin_dark', 'skin_avg', 'meanIAT', 'meanExp', 'age', 'height_imputed', 'weight_imputed']
    # Ensure numeric columns exist in data; if missing, create with NaN
    for col in numeric_cols:
        if col not in data.columns:
            data[col] = np.nan
    X_parts = [
        data[numeric_cols].astype(float),
        pos_dummies,
        league_dummies
    ]
    X = pd.concat(X_parts, axis=1)

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Response
    y = pd.to_numeric(data['redCards'], errors='coerce').astype(float)

    # Offset: log(games). Make sure games > 0 (should be true after transform)
    offset = np.log(pd.to_numeric(data['games'], errors='coerce').astype(float))

    # Fit a Negative Binomial GLM with offset to model counts with overdispersion.
    # Use statsmodels' GLM with NegativeBinomial family. Then compute cluster-robust SEs by referee (refNum).
    try:
        model_glm = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
        result = model_glm.fit(maxiter=100, disp=False)
    except Exception:
        # Fallback to Poisson with robust SEs if NegativeBinomial fails to converge
        model_glm = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
        result = model_glm.fit(maxiter=100, disp=False)

    # Obtain cluster-robust covariance (by referee id) to account for non-independence of dyads judged by the same referee
    # If refNum isn't present or has NA or only one unique group, fall back to HC1
    robust_cov = None
    if 'refNum' in data.columns:
        groups = data['refNum']
        try:
            if groups.notnull().all() and groups.nunique() > 1:
                robust_cov = cov_cluster(result, groups)
            else:
                # Not suitable for clustering
                robust_cov = cov_hc1(result)
        except Exception:
            robust_cov = cov_hc1(result)
    else:
        robust_cov = cov_hc1(result)

    # Build a lightweight result-like object that exposes params, bse, tvalues, pvalues, cov_params, and summary()
    class RobustResults:
        def __init__(self, orig_result, cov_matrix):
            self.orig = orig_result
            self.params = pd.Series(orig_result.params, index=orig_result.params.index.copy())
            self.cov = pd.DataFrame(cov_matrix, index=self.params.index, columns=self.params.index)
            self.bse = pd.Series(np.sqrt(np.diag(self.cov.values)), index=self.params.index)
            self.tvalues = self.params / self.bse
            # two-sided z-test p-values
            self.pvalues = pd.Series(2 * (1 - stats.norm.cdf(np.abs(self.tvalues))), index=self.params.index)

        def cov_params(self):
            return self.cov

        def summary(self):
            # Return a simple table similar to standard summaries
            tbl = pd.DataFrame({
                'coef': self.params,
                'std_err': self.bse,
                'z': self.tvalues,
                'P>|z|': self.pvalues
            })
            # Use orig.summary() as well if available, but preds use robust covariances above.
            try:
                orig_sum = self.orig.summary().as_text()
            except Exception:
                orig_sum = None
            # Build a combined string
            header = "Robust results (cluster-robust / HC1 adjusted SEs)\n"
            table_str = tbl.to_string(float_format=lambda x: f"{x:0.4f}")
            if orig_sum:
                return header + table_str + "\n\nOriginal model summary:\n" + orig_sum
            else:
                return header + table_str

    return RobustResults(result, robust_cov)