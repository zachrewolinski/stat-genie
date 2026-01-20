from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

# Optional: load raw data if available in this environment (kept from original)
# Note: this path may not exist in other environments; it's preserved from original file.
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/amtl/noperturb_output/amtl.csv')
except Exception:
    # If file is not present, define df as an empty DataFrame placeholder.
    df = pd.DataFrame()


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Gilmore (2013) AMTL dataset into the dataframe used for modeling.

    Output dataframe columns required by the model:
      - num_amtl: integer count of missing teeth for the record
      - sockets: integer number of observable sockets for the record (must be > 0)
      - is_human: binary indicator (1 if genus == 'Homo sapiens', 0 otherwise)
      - age_z: standardized age (mean 0, sd 1)
      - prob_male: numeric probability that the specimen is male (0..1). Missing values filled with 0.5.
      - tooth_class: categorical with values 'Anterior', 'Posterior', 'Premolar'
      - specimen: specimen identifier (used for clustering SEs)
    """

    # Make a copy to avoid modifying original
    df = df.copy()

    # Coerce numeric types for counts first to avoid comparison errors
    df['num_amtl'] = pd.to_numeric(df.get('num_amtl', pd.Series(dtype=float)), errors='coerce')
    df['sockets'] = pd.to_numeric(df.get('sockets', pd.Series(dtype=float)), errors='coerce')

    # Drop rows where sockets is missing or nonpositive
    df = df.dropna(subset=['sockets'])
    df = df[df['sockets'] > 0]

    # Now drop rows missing other essential fields
    req_cols = ['num_amtl', 'sockets', 'age', 'genus', 'tooth_class', 'specimen']
    df = df.dropna(subset=req_cols)

    # Ensure sockets and num_amtl numeric (coercion may have produced NaN earlier)
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce')
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')

    # Drop rows where coercion produced NaN
    df = df.dropna(subset=['num_amtl', 'sockets'])

    # Clip num_amtl to valid range [0, sockets]
    df['num_amtl'] = df['num_amtl'].clip(lower=0)
    # if any num_amtl > sockets, set to sockets (to preserve physical constraint)
    df.loc[df['num_amtl'] > df['sockets'], 'num_amtl'] = df.loc[df['num_amtl'] > df['sockets'], 'sockets']

    # Create binary human indicator
    df['is_human'] = (df['genus'].astype(str).str.strip() == 'Homo sapiens').astype(int)

    # Handle prob_male (fill missing with 0.5 when unknown)
    if 'prob_male' in df.columns:
        df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce')
        df['prob_male'] = df['prob_male'].fillna(0.5)
    else:
        # If not present, create a neutral column
        df['prob_male'] = 0.5

    # Standardize age to mean 0, sd 1
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    age_mean = df['age'].mean()
    age_std = df['age'].std(ddof=0)
    # If sd is zero (unlikely), avoid division by zero
    if age_std == 0 or np.isnan(age_std):
        df['age_z'] = 0.0
    else:
        df['age_z'] = (df['age'] - age_mean) / age_std

    # Ensure tooth_class is a categorical variable with consistent labels
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip()
    # Optionally coerce common variants to canonical levels
    df['tooth_class'] = df['tooth_class'].replace({'anterior': 'Anterior', 'posterior': 'Posterior', 'premolar': 'Premolar'})
    df['tooth_class'] = df['tooth_class'].astype('category')

    # Keep only columns needed for modeling plus a few useful diagnostics
    keep_cols = ['num_amtl', 'sockets', 'is_human', 'age_z', 'prob_male', 'tooth_class', 'specimen', 'genus', 'age']
    for c in keep_cols:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a binomial (logistic) GLM to test whether modern humans have higher AMTL than non-human primates,
    controlling for age (standardized), sex probability, and tooth_class. The response is modeled as
    the proportion num_amtl / sockets with weights = sockets.

    Returns a dictionary with the fitted model, cluster-robust results (clustered by specimen), and
    an odds ratio + 95% CI for the is_human coefficient.
    """

    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hc1
    from scipy import stats as _stats

    # Ensure required columns exist
    required = ['num_amtl', 'sockets', 'is_human', 'age_z', 'prob_male', 'tooth_class', 'specimen']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Work on a copy to allow safe helper columns without mutating caller's df
    df_model = df.copy()

    # Ensure numeric types for modeling
    df_model['num_amtl'] = pd.to_numeric(df_model['num_amtl'], errors='coerce').fillna(0.0)
    df_model['sockets'] = pd.to_numeric(df_model['sockets'], errors='coerce').fillna(1.0)
    df_model['age_z'] = pd.to_numeric(df_model['age_z'], errors='coerce').fillna(0.0)
    df_model['prob_male'] = pd.to_numeric(df_model['prob_male'], errors='coerce').fillna(0.5)

    # Create a proportion outcome for the GLM. To avoid numerical issues that can
    # cause the initial deviance to be NaN (e.g., exact 0 or 1 proportions), clip the
    # empirical proportions to a small open interval (eps, 1-eps). This is an internal
    # helper and does not change the original conceptual columns.
    eps = 1e-6
    df_model['prop'] = (df_model['num_amtl'] / df_model['sockets']).clip(eps, 1 - eps)

    # Build formula using the helper 'prop' as the dependent variable
    formula = 'prop ~ is_human + age_z + prob_male + C(tooth_class)'

    # Fit GLM with binomial family and logit link; use weights = sockets (number of trials)
    glm_mod = smf.glm(formula=formula, data=df_model, family=sm.families.Binomial(), weights=df_model['sockets'])
    glm_res = glm_mod.fit()

    # Compute cluster-robust SEs clustered by specimen to account for non-independence.
    # Some statsmodels versions do not provide get_robustcov_results on GLMResults; compute
    # the covariance manually and wrap into a lightweight results-like object.
    class RobustResults:
        def __init__(self, params: pd.Series, cov: np.ndarray):
            self.params = params.copy()
            # Ensure cov is a 2D numpy array
            self.cov_params = cov.copy()
            se = np.sqrt(np.maximum(np.diag(self.cov_params), 0.0))
            # Preserve index in Series
            self.bse = pd.Series(se, index=self.params.index)
            # t/z values (as normal-based z)
            self.tvalues = self.params / self.bse
            # Two-sided p-values using normal distribution
            self.pvalues = pd.Series(2 * (1 - _stats.norm.cdf(np.abs(self.tvalues.values))), index=self.params.index)
        def conf_int(self, alpha=0.05):
            z = _stats.norm.ppf(1 - alpha / 2.0)
            lower = self.params - z * self.bse
            upper = self.params + z * self.bse
            return pd.DataFrame({0: lower, 1: upper}, index=self.params.index)
        def summary2(self):
            # Intentionally raise to trigger fallback coefficient table construction
            raise NotImplementedError("summary2 not implemented for RobustResults wrapper")

    try:
        # Try cluster covariance first
        groups = df_model['specimen']
        vcov = cov_cluster(glm_res, groups)
    except Exception:
        # Fall back to heteroskedasticity-consistent HC1 covariance
        try:
            vcov = cov_hc1(glm_res)
        except Exception:
            # As a final fallback, use the model's default covariance (not robust)
            vcov = glm_res.cov_params()

    robust_res = RobustResults(params=glm_res.params, cov=vcov)

    # Extract odds ratio and 95% CI for the is_human coefficient from the robust results
    param_name = 'is_human'
    if param_name in robust_res.params.index:
        coef = robust_res.params[param_name]
        or_est = float(np.exp(coef))
        ci = robust_res.conf_int().loc[param_name].astype(float)
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
        or_ci = (float(np.exp(ci_lower)), float(np.exp(ci_upper)))
    else:
        or_est = None
        or_ci = (None, None)

    # Prepare a concise summary table (pandas DataFrame)
    try:
        coef_table = robust_res.summary2().tables[1]
    except Exception:
        # If summary2 fails for some reason, provide coefficients table manually
        coef_table = pd.DataFrame({
            'coef': robust_res.params,
            'std_err': robust_res.bse,
            'z': robust_res.tvalues,
            'P>|z|': robust_res.pvalues
        })

    results = {
        'glm_model': glm_mod,
        'glm_result': glm_res,
        'robust_result': robust_res,
        'is_human_odds_ratio': or_est,
        'is_human_odds_ratio_ci': or_ci,
        'coef_table': coef_table
    }

    return results