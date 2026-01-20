from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/replace_with_rvs_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dyad-level dataset to produce the columns required for modeling.

    Produces:
    - avg_rater: average of rater1 and rater2 (0-1 normalized scale)
    - SkinToneCategory: 'Light' (avg <= 0.25), 'Dark' (avg >= 0.75), 'Other' otherwise
    - SkinDark: binary 1 for 'Dark', 0 for 'Light' (rows with 'Other' removed)
    - age: age in years at reference date (season midpoint)
    - meanIAT_z: z-scored meanIAT across the kept rows
    - ensures required columns are numeric and drops rows with missing required information
    """
    df = df.copy()

    # Keep relevant columns first to reduce memory and accidental passing of unused columns
    required_cols = ['redCards', 'games', 'rater1', 'rater2', 'birthday', 'meanIAT', 'weight', 'height', 'position', 'leagueCountry', 'refNum']
    # If any of the required columns are missing in input, let it fail clearly
    missing_cols = [c for c in required_cols if c not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Input dataframe is missing required columns: {missing_cols}")

    # Convert rater columns to numeric and compute average rater score
    df['rater1'] = pd.to_numeric(df['rater1'], errors='coerce')
    df['rater2'] = pd.to_numeric(df['rater2'], errors='coerce')
    df['avg_rater'] = df[['rater1', 'rater2']].mean(axis=1)

    # Define thresholds for 'light' and 'dark'. The rater variables were normalized to a 0-1 scale
    # corresponding to a 5-point scale. Here we take the extremes to form a clean contrast:
    # Light: avg_rater <= 0.25   (roughly ratings near the lightest category)
    # Dark:  avg_rater >= 0.75   (roughly ratings near the darkest category)
    df['SkinToneCategory'] = pd.cut(df['avg_rater'], bins=[-0.01, 0.25, 0.75, 1.01], labels=['Light', 'Other', 'Dark'])

    # Keep only clear 'Light' and 'Dark' cases to directly answer the research question
    df = df[df['SkinToneCategory'].isin(['Light', 'Dark'])].copy()

    # Create the binary independent variable: 1 = Dark, 0 = Light
    df['SkinDark'] = (df['SkinToneCategory'] == 'Dark').astype(int)

    # Ensure numeric columns are numeric
    df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce').fillna(0).astype(int)
    df['games'] = pd.to_numeric(df['games'], errors='coerce')
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df['height'] = pd.to_numeric(df['height'], errors='coerce')

    # Parse birthday and compute age at a reference date (season midpoint). Use 2013-01-01 or mid-year.
    # We'll choose 2013-06-01 as a mid-season reference for the 2012-2013 season.
    df['birthday_parsed'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y', errors='coerce')
    reference_date = pd.to_datetime('2013-06-01')
    df['age'] = (reference_date - df['birthday_parsed']).dt.days / 365.25

    # Drop rows with missing critical variables: games (must be >=1), redCards (we already coerced NA to 0), age, avg_rater
    df = df.dropna(subset=['games', 'age', 'avg_rater', 'meanIAT'])

    # Remove rows with non-positive games
    df = df[df['games'] >= 1].copy()

    # Standardize meanIAT for model interpretability
    mean_iat_std = df['meanIAT'].std(ddof=0)
    if pd.isna(mean_iat_std) or mean_iat_std == 0:
        df['meanIAT_z'] = 0.0
    else:
        df['meanIAT_z'] = (df['meanIAT'] - df['meanIAT'].mean()) / mean_iat_std

    # Ensure categorical columns are strings / categories for formula handling
    df['position'] = df['position'].astype('category')
    df['leagueCountry'] = df['leagueCountry'].astype('category')

    # Keep only the columns needed for modeling and diagnostics
    keep_cols = ['playerShort', 'player', 'club', 'leagueCountry', 'position', 'height', 'weight', 'games', 'redCards', 'photoID', 'rater1', 'rater2', 'avg_rater', 'SkinToneCategory', 'SkinDark', 'birthday_parsed', 'age', 'meanIAT', 'meanIAT_z', 'refNum']
    existing_keep = [c for c in keep_cols if c in df.columns]
    df = df[existing_keep].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative binomial regression predicting redCards counts with log(games) as an offset.
    The primary coefficient of interest is on SkinDark (1 = dark, 0 = light).

    Controls: meanIAT_z (standardized country implicit bias), age, weight, height,
    categorical controls for position and leagueCountry. Cluster-robust standard errors by refNum.

    Returns a dictionary with the robust results object and a table of incidence rate ratios (IRRs).
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    import numpy as np
    from statsmodels.stats.sandwich_covariance import cov_cluster

    # Ensure required columns exist
    for c in ['redCards', 'games', 'SkinDark', 'meanIAT_z', 'age', 'weight', 'height', 'position', 'leagueCountry', 'refNum']:
        if c not in df.columns:
            raise ValueError(f"Required column for modeling missing: {c}")

    # Build formula. Use categorical encoding for position and leagueCountry within the formula using C(...)
    formula = 'redCards ~ SkinDark + meanIAT_z + age + weight + height + C(position) + C(leagueCountry)'

    # Fit a GLM negative binomial with offset = log(games)
    # Handle any zero or negative games earlier in transform; still guard here
    offset = np.log(df['games'].astype(float))

    glm_nb = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=offset)
    res = glm_nb.fit()

    # Compute cluster-robust covariance clustered on refNum. If clustering fails, fall back to HC1 via get_robustcov_results.
    try:
        groups = df['refNum'].values
        cov = cov_cluster(res, groups)
    except Exception:
        try:
            robust_wrapper = res.get_robustcov_results(cov_type='HC1')
            cov = robust_wrapper.cov_params()
        except Exception:
            # As a last resort, use the model's default covariance matrix
            cov = res.cov_params()

    # Prepare a simple results-like object that exposes params and conf_int()
    class RobustResults:
        def __init__(self, params: pd.Series, cov: np.ndarray):
            self.params = params
            self._cov = cov
            self.bse = pd.Series(np.sqrt(np.diag(cov)), index=params.index)

        def conf_int(self, alpha=0.05):
            # two-sided (1-alpha) CI
            z = np.abs(scipy.stats.norm.ppf(alpha / 2))
            lower = self.params - z * self.bse
            upper = self.params + z * self.bse
            return pd.DataFrame({0: lower, 1: upper}, index=self.params.index)

        def cov_params(self):
            return self._cov

    robust_res = RobustResults(res.params, cov)

    # Compute incidence rate ratios (IRR) and 95% CIs from robust estimates
    params = robust_res.params
    conf = robust_res.conf_int()
    irr = np.exp(params)
    irr_ci_lower = np.exp(conf[0])
    irr_ci_upper = np.exp(conf[1])

    irr_table = (pd.DataFrame({'IRR': irr, 'IRR_ci_lower': irr_ci_lower, 'IRR_ci_upper': irr_ci_upper})
                 .loc[params.index])

    # Return a dictionary with the robust results object and IRR table for interpretation
    return {
        'robust_results': robust_res,
        'irr_table': irr_table
    }