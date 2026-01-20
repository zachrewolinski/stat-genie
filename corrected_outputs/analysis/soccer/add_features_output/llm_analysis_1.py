from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/add_features_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dyad-level data into the analysis dataframe. Produces:
      - mean_skin_rater: mean of rater1 and rater2 (0..1 normalized ratings)
      - SkinGroup: 'Dark', 'Light', or 'Intermediate' (based on thresholds)
      - dark_skin: binary 1 if Dark, 0 if Light (we drop 'Intermediate' rows)
      - ensures redCards and games are integers and drops dyads with games <= 0
    """
    df = df.copy()

    # Ensure numeric columns exist and coerce types where necessary
    for col in ['rater1', 'rater2', 'redCards', 'games', 'age', 'height', 'weight', 'meanIAT', 'meanExp', 'refNum']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute mean skin rating from the two raters (normalized 0..1 in source)
    # If rater columns are missing, this will produce NaN and later be filtered out
    if {'rater1', 'rater2'}.issubset(df.columns):
        df['mean_skin_rater'] = df[['rater1', 'rater2']].mean(axis=1)
    else:
        df['mean_skin_rater'] = pd.NA

    # Create coarse skin group categories to compare 'Dark' vs 'Light'
    # Thresholds: <=0.4 -> Light, >=0.6 -> Dark, between -> Intermediate
    def skin_group(x):
        if pd.isna(x):
            return pd.NA
        try:
            if x <= 0.4:
                return 'Light'
            if x >= 0.6:
                return 'Dark'
            return 'Intermediate'
        except Exception:
            return pd.NA

    df['SkinGroup'] = df['mean_skin_rater'].apply(skin_group)

    # Keep only clear Light vs Dark comparisons to match research question
    df = df[df['SkinGroup'].isin(['Light', 'Dark'])].copy()

    # Binary treatment variable: dark_skin = 1 if Dark, 0 if Light
    df['dark_skin'] = (df['SkinGroup'] == 'Dark').astype(int)

    # Ensure redCards is integer count and games as exposure (must be positive)
    if 'redCards' in df.columns:
        df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce').fillna(0).astype(int)
    else:
        df['redCards'] = 0

    if 'games' in df.columns:
        df['games'] = pd.to_numeric(df['games'], errors='coerce')
    else:
        df['games'] = pd.NA

    # Drop rows where games is missing or non-positive because we cannot compute exposure offset
    df = df.dropna(subset=['games'])
    df = df[df['games'] > 0].copy()

    # Clean categorical covariates
    if 'position' in df.columns:
        df['position'] = df['position'].astype('category')
    if 'playerShort' in df.columns:
        df['playerShort'] = df['playerShort'].astype('category')

    # Final columns that will be used in modeling are kept; others preserved but not required
    required_cols = [
        'playerShort', 'refNum', 'mean_skin_rater', 'SkinGroup', 'dark_skin',
        'redCards', 'games', 'position', 'age', 'height', 'weight', 'meanIAT', 'meanExp'
    ]

    # Make sure all required columns exist in the returned dataframe; if missing, create as NA
    for c in required_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # Return dataframe filtered and typed
    return df[required_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a negative binomial regression for red card counts with exposure offset = log(games).
    Returns:
      - fit: the fitted GLM results (Negative Binomial)
      - clustered_results: fit-like object with cluster-robust SEs at referee (refNum) level
      - irr: incident rate ratios (exp(coef))
      - conf_int_irr: exponentiated confidence intervals for IRRs

    Model formula:
      redCards ~ dark_skin + age + height + weight + C(position) + meanIAT + meanExp
    Offset: log(games)
    Clustered SEs: clustered by refNum to account for within-referee dependence.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.sandwich_covariance import cov_cluster
    from scipy import stats as sstats

    # Copy to avoid modifying input
    df = df.copy()

    # Drop rows missing key model variables
    df = df.dropna(subset=['redCards', 'dark_skin', 'games', 'refNum'])

    # Replace any remaining NA numeric covariates with column means (simple, transparent imputation)
    for col in ['age', 'height', 'weight', 'meanIAT', 'meanExp']:
        if col in df.columns:
            if df[col].isna().any():
                # If all values are NA, keep them as NA; otherwise fill with mean of available values
                if df[col].notna().any():
                    df[col] = df[col].fillna(df[col].mean())

    # Build formula: categorical position is modeled with C(position)
    formula = 'redCards ~ dark_skin + age + height + weight + C(position) + meanIAT + meanExp'

    # Fit a GLM Negative Binomial with log(games) as offset
    glm_nb = smf.glm(formula=formula,
                     data=df,
                     family=sm.families.NegativeBinomial(),
                     offset=np.log(df['games']))
    fit = glm_nb.fit()

    # Compute clustered (referee-level) robust covariance results
    # Use statsmodels' sandwich covariance estimator for clustering
    # cov_cluster accepts the fitted results and group labels
    try:
        clustered_cov = cov_cluster(fit, df['refNum'])
    except Exception:
        # If cov_cluster fails for any reason, fall back to the model's default covariance
        clustered_cov = fit.cov_params()

    # Create a lightweight wrapper object that mimics the relevant parts of a results instance
    class ClusteredResults:
        def __init__(self, fit_res, cov_matrix):
            self._fit = fit_res
            self.params = fit_res.params.copy()
            self.cov = cov_matrix
            # Protect against malformed covariance matrices
            try:
                self.bse = np.sqrt(np.diag(self.cov))
            except Exception:
                self.bse = fit_res.bse.copy()

        def conf_int(self, alpha=0.05):
            z = sstats.norm.ppf(1 - alpha / 2)
            lower = self.params - z * self.bse
            upper = self.params + z * self.bse
            return pd.DataFrame({0: lower, 1: upper}, index=self.params.index)

        def summary(self):
            # Return the original summary object; it has an as_text() method.
            # Note: this summary will reflect original (non-clustered) SEs in the printed table,
            # but the clustered covariance, bse, and conf_int provided by this wrapper should be used
            # for inference that relies on clustering.
            return self._fit.summary()

    clustered = ClusteredResults(fit, clustered_cov)

    # Incident rate ratios (IRR) and exponentiated confidence intervals from the clustered results
    params = clustered.params
    irr = np.exp(params)
    conf = clustered.conf_int()
    conf_int_irr = np.exp(conf)
    conf_int_irr.columns = ['IRR_ci_lower', 'IRR_ci_upper']

    # Pack results
    results = {
        'fit': fit,
        'clustered_results': clustered,
        'irr': irr,
        'conf_int_irr': conf_int_irr,
        'summary_text': clustered.summary().as_text()
    }

    return results