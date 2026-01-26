from typing import Any
import numpy as np
import pandas as pd
import sklearn  # noqa: F401
import scipy  # noqa: F401
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt  # noqa: F401
import pickle  # noqa: F401


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/noperturb_output/soccer.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis dataframe. Creates skin-tone measures, age, categorical skin groups, and filters to dyads with non-missing required fields and games>0.

    Final dataframe columns required for modeling (and created if necessary):
      - SkinToneAvg (float): average of rater1 and rater2
      - SkinToneCat (category): 'Light' | 'Medium' | 'Dark'
      - SkinDark (int): 1 if 'Dark', 0 if 'Light' (we will subset to Light vs Dark for primary analysis)
      - Age (float): age in years at reference date
      - redCards (int): outcome (keeps original column name)
      - games (int): exposure variable
      - meanIAT, meanExp, position, height, weight, goals, yellowCards, refNum, leagueCountry (kept from original)
    """
    # Work on a copy
    df = df.copy()

    # Ensure rater columns exist and drop rows missing rater scores or outcome/exposure
    required = ['rater1', 'rater2', 'redCards', 'games']
    df = df.dropna(subset=required)

    # Compute average skin tone (rater values are normalized to 1 in source)
    df['SkinToneAvg'] = (df['rater1'].astype(float) + df['rater2'].astype(float)) / 2.0

    # Create categorical skin tone (Light, Medium, Dark) using cutpoints on the 0-1 normalized scale
    # Boundaries: Light <= 0.33, Medium (0.33,0.66], Dark > 0.66
    def _skin_cat(x):
        if pd.isna(x):
            return pd.NA
        if x <= 0.33:
            return 'Light'
        elif x > 0.66:
            return 'Dark'
        else:
            return 'Medium'

    df['SkinToneCat'] = df['SkinToneAvg'].apply(_skin_cat).astype('category')

    # Create binary indicator used in the primary analysis: 1 for Dark, 0 for Light
    # (Rows with 'Medium' remain and will be filtered out for the main comparison below)
    df['SkinDark'] = df['SkinToneCat'].apply(lambda s: 1 if s == 'Dark' else (0 if s == 'Light' else pd.NA))

    # Parse birthday to compute Age (use mid-season reference date: 2013-01-01)
    # birthday column uses dd.mm.yyyy format; parse robustly
    df['birthday_parsed'] = pd.to_datetime(df['birthday'], dayfirst=True, errors='coerce')
    ref_date = pd.to_datetime('2013-01-01')
    df['Age'] = ((ref_date - df['birthday_parsed']).dt.days / 365.25).astype(float)

    # Ensure numeric columns used as covariates are numeric
    num_cols = ['height', 'weight', 'goals', 'yellowCards', 'redCards', 'games', 'meanIAT', 'meanExp']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Remove dyads with zero or missing games (cannot use as exposure)
    df = df[df['games'].notna() & (df['games'] > 0)]

    # Keep only rows that have SkinDark defined (i.e., Light or Dark); this focuses the primary comparison
    df = df[df['SkinDark'].notna()]

    # Ensure SkinDark is integer 0/1
    try:
        df['SkinDark'] = df['SkinDark'].astype(int)
    except Exception:
        # fallback: cast via float then int
        df['SkinDark'] = df['SkinDark'].astype(float).astype(int)

    # Ensure categorical variables have appropriate dtype
    if 'position' in df.columns:
        df['position'] = df['position'].astype('category')
    if 'leagueCountry' in df.columns:
        df['leagueCountry'] = df['leagueCountry'].astype('category')

    # Final check: drop rows with missing values on key covariates used in the model
    model_vars = ['redCards', 'games', 'SkinDark', 'meanIAT', 'meanExp', 'position', 'Age', 'height', 'weight', 'goals', 'yellowCards', 'refNum', 'leagueCountry']
    present_vars = [v for v in model_vars if v in df.columns]
    df = df.dropna(subset=present_vars)

    # Reset index and return
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative binomial regression of redCards on SkinDark with games as an offset (log exposure).
    Uses robust clustered standard errors clustered by referee (refNum).

    Returns the clustered-results object (statsmodels-like object) for inspection along with other useful items.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    import numpy as np
    from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hc3
    from scipy.stats import norm
    from types import SimpleNamespace

    # Work on a copy
    data = df.copy()

    # Ensure required columns exist (should be satisfied if transform was used)
    required = ['redCards', 'games', 'SkinDark', 'meanIAT', 'meanExp', 'position', 'Age', 'height', 'weight', 'goals', 'yellowCards', 'refNum', 'leagueCountry']
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Formula: primary predictor is SkinDark (1=Dark, 0=Light). Include controls.
    # We include leagueCountry and position as categorical fixed effects.
    formula = (
        'redCards ~ SkinDark + meanIAT + meanExp + C(position) + Age + height + weight + goals + yellowCards + C(leagueCountry)'
    )

    # Create offset = log(games)
    data['offset_log_games'] = np.log(data['games'].astype(float))

    # Fit negative binomial via GLM with NegativeBinomial family
    model_nb = smf.glm(formula=formula, data=data, family=sm.families.NegativeBinomial(), offset=data['offset_log_games'])
    res_nb = model_nb.fit()

    # Obtain robust clustered standard errors clustered by referee ID (refNum)
    clustered_cov = None
    try:
        groups = np.asarray(data['refNum'])
        clustered_cov = cov_cluster(res_nb, groups)
    except Exception:
        # If clustering fails, fall back to HC3 robust covariance
        clustered_cov = cov_hc3(res_nb)

    # Build a lightweight results-like object that exposes params and conf_int()
    class ClusteredResults:
        def __init__(self, base_res, cov):
            self._base = base_res
            self.params = base_res.params.copy()
            self.cov_params = cov
            self.bse = np.sqrt(np.diag(self.cov_params))
            self._z = norm.ppf(0.975)

        def conf_int(self):
            lower = self.params - self._z * self.bse
            upper = self.params + self._z * self.bse
            ci = pd.concat([lower, upper], axis=1)
            ci.columns = [0, 1]
            return ci

        def summary(self):
            return self._base.summary()

        # Expose other commonly used attributes as needed
        def __getattr__(self, item):
            return getattr(self._base, item)

    res_nb_clustered = ClusteredResults(res_nb, clustered_cov)

    # For convenience, also compute and attach incidence rate ratios (IRRs) and CIs
    params = res_nb_clustered.params
    conf = res_nb_clustered.conf_int()
    irr = np.exp(params)
    irr_ci = np.exp(conf)
    irr_df = (irr.rename('IRR').to_frame()).join(irr_ci)
    irr_df.columns = ['IRR', 'IRR_ci_lower', 'IRR_ci_upper']

    # Package results in a dictionary for easy programmatic access
    results = {
        'model_result': res_nb,
        'clustered_result': res_nb_clustered,
        'irr_table': irr_df,
        'formula': formula
    }

    return results