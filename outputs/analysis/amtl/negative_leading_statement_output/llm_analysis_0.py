from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hc3
from scipy import stats as _stats

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/negative_leading_statement_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a cleaned dataframe ready for binomial regression.

    Output columns required for modeling (kept or created):
      - num_amtl: integer count of missing teeth for the record (as given)
      - sockets: integer count of observable sockets (as given)
      - prop_amtl: proportion num_amtl / sockets (used as GLM endog with weights=sockets)
      - Is_Human: binary indicator (1 if genus == 'Homo sapiens', else 0)
      - age_std: standardized age (mean 0, sd 1)
      - prob_male: probability specimen is male (0-1)
      - tooth_class: categorical with levels preserved (Anterior, Premolar, Posterior)
      - specimen: specimen identifier (used for clustering SEs)
    """
    df = df.copy()

    # Drop rows missing critical fields
    required_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class', 'specimen']
    df = df.dropna(subset=required_cols)

    # Ensure sockets is numeric and positive; remove impossible rows
    df['sockets'] = pd.to_numeric(df['sockets'], errors='coerce')
    df = df[df['sockets'] >= 1]
    df['sockets'] = df['sockets'].astype(int)

    # Ensure num_amtl is integer, non-negative, and not larger than sockets
    df['num_amtl'] = pd.to_numeric(df['num_amtl'], errors='coerce').fillna(0).astype(int)
    # cap num_amtl at sockets (some datasets can contain small scoring inconsistencies)
    mask_too_large = df['num_amtl'] > df['sockets']
    df.loc[mask_too_large, 'num_amtl'] = df.loc[mask_too_large, 'sockets']
    df.loc[df['num_amtl'] < 0, 'num_amtl'] = 0

    # Create proportion column for GLM with weights
    # Avoid division by zero (sockets already filtered to >=1)
    df['prop_amtl'] = df['num_amtl'] / df['sockets']

    # Create binary human indicator
    df['Is_Human'] = (df['genus'].astype(str).str.strip() == 'Homo sapiens').astype(int)

    # Standardize age to improve model fitting/interpretation
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    age_mean = df['age'].mean()
    age_std = df['age'].std(ddof=0) if df['age'].std(ddof=0) > 0 else 1.0
    df['age_std'] = (df['age'] - age_mean) / age_std

    # Ensure prob_male is numeric and bound to [0,1]
    df['prob_male'] = pd.to_numeric(df['prob_male'], errors='coerce').fillna(0.5)
    df.loc[df['prob_male'] < 0, 'prob_male'] = 0.0
    df.loc[df['prob_male'] > 1, 'prob_male'] = 1.0

    # Ensure tooth_class is categorical and has consistent labels
    df['tooth_class'] = df['tooth_class'].astype(str).str.strip().replace({
        'premolar': 'Premolar', 'anterior': 'Anterior', 'posterior': 'Posterior'
    })
    df['tooth_class'] = pd.Categorical(df['tooth_class'], categories=['Anterior', 'Premolar', 'Posterior'])

    # Keep only necessary columns (but keep original useful metadata)
    keep_cols = ['specimen', 'genus', 'Is_Human', 'num_amtl', 'sockets', 'prop_amtl', 'age', 'age_std', 'prob_male', 'tooth_class', 'pop']
    for c in keep_cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM to test whether modern humans (Is_Human==1) have higher AMTL than non-human primates,
    controlling for age, sex probability, and tooth class. Uses observation-level binomial weights (sockets)
    and clusters standard errors by specimen.

    Model form (proportion response, weights = sockets):
      prop_amtl ~ Is_Human + age_std + prob_male + C(tooth_class)

    Returns cluster-robust results (clustered by specimen).
    """
    import statsmodels.api as _sm
    import statsmodels.formula.api as _smf

    # Drop any rows with missing pieces required for modeling
    model_df = df.dropna(subset=['prop_amtl', 'sockets', 'Is_Human', 'age_std', 'prob_male', 'tooth_class', 'specimen']).copy()

    # Formula: proportion response with observation weights = number of trials (sockets)
    formula = 'prop_amtl ~ Is_Human + age_std + prob_male + C(tooth_class)'

    # Fit binomial GLM on proportions with weights = sockets
    glm_res = _smf.glm(formula=formula, data=model_df, family=_sm.families.Binomial(), weights=model_df['sockets']).fit()

    # Attempt to produce cluster-robust covariance (cluster by specimen).
    # Some statsmodels result objects may not provide get_robustcov_results; compute covariance manually if needed.
    robust_res = None
    try:
        # Prefer using the built-in method if available
        get_robust = getattr(glm_res, 'get_robustcov_results', None)
        if callable(get_robust):
            try:
                robust_res = glm_res.get_robustcov_results(cov_type='cluster', groups=model_df['specimen'])
            except Exception:
                # Fall back to HC3 via built-in if cluster fails
                robust_res = glm_res.get_robustcov_results(cov_type='HC3')
        else:
            raise AttributeError("get_robustcov_results not available")
    except Exception:
        # Manual construction of a proxy results object with cluster-robust or HC3 covariance
        try:
            cov = cov_cluster(glm_res, model_df['specimen'])
        except Exception:
            cov = cov_hc3(glm_res)

        params = glm_res.params
        bse = pd.Series(np.sqrt(np.diag(cov)), index=params.index)
        # two-sided p-values using normal approximation
        z_stats = params / bse
        pvalues = pd.Series(2 * _stats.norm.sf(np.abs(z_stats)), index=params.index)

        crit = float(_stats.norm.ppf(0.975))
        lower = params - crit * bse
        upper = params + crit * bse
        conf_int_df = pd.DataFrame({0: lower, 1: upper})

        class RobustResultsProxy:
            def __init__(self, orig_res, params, bse, pvalues, conf_int_df, cov):
                self._orig = orig_res
                self.params = params
                self.bse = bse
                self.pvalues = pvalues
                self._conf_int = conf_int_df
                self.cov_params = cov

            def conf_int(self):
                return self._conf_int

        robust_res = RobustResultsProxy(glm_res, params, bse, pvalues, conf_int_df, cov)

    # Compute and attach a simple summary for the primary contrast (Is_Human): coefficient, se, z, p, 95% CI, odds-ratio
    coef = robust_res.params.get('Is_Human', None)
    se = robust_res.bse.get('Is_Human', None)
    if coef is not None and se is not None:
        z = coef / se
        pval = float(robust_res.pvalues.get('Is_Human'))
        ci_lower, ci_upper = robust_res.conf_int().loc['Is_Human']
        # For binomial-logit link, exponentiate coefficient to get odds ratio
        or_est = float(np.exp(coef))
        or_ci = (float(np.exp(ci_lower)), float(np.exp(ci_upper)))
        summary_small = {
            'coef_Is_Human': float(coef),
            'se_Is_Human': float(se),
            'z_Is_Human': float(z),
            'p_Is_Human': float(pval),
            'ci95_Is_Human': [float(ci_lower), float(ci_upper)],
            'odds_ratio_Is_Human': or_est,
            'odds_ratio_ci95': list(or_ci)
        }
    else:
        summary_small = None

    # Return a dict with the fitted object and a concise summary for convenient programmatic checks
    return {
        'model_fit': robust_res,
        'summary_Is_Human': summary_small,
        'formula': formula,
        'n_obs': int(model_df.shape[0]),
        'n_specimens': int(model_df['specimen'].nunique())
    }