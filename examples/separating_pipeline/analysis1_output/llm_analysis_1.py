from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
from statsmodels.stats.sandwich_covariance import cov_hc3


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure key numeric columns are numeric where possible
    numeric_cols = ['masfem', 'masfem_mturk', 'min', 'wind', 'category', 'alldeaths', 'ndam15', 'ndam', 'elapsedyrs', 'year']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Create primary dependent variable: Deaths (keep original counts)
    # If alldeaths contains NaN, leave as NaN so model can drop those rows as needed
    if 'alldeaths' in df.columns:
        df['Deaths'] = df['alldeaths']
    else:
        df['Deaths'] = np.nan

    # Create log-transformed damage variable (secondary outcome): log(1 + damage in 2015 dollars)
    if 'ndam15' in df.columns:
        df['log_damage'] = np.log1p(df['ndam15'])
    else:
        df['log_damage'] = np.nan

    # Standardize (z-score) the masfem measure to aid interpretation
    if 'masfem' in df.columns:
        mas_mean = df['masfem'].mean(skipna=True)
        mas_std = df['masfem'].std(skipna=True)
        if pd.isna(mas_std) or mas_std == 0:
            df['masfem_z'] = df['masfem'] - mas_mean
        else:
            df['masfem_z'] = (df['masfem'] - mas_mean) / mas_std
    else:
        df['masfem_z'] = np.nan

    # Also z-score the MTurk rating if available (alternative IV / robustness check)
    if 'masfem_mturk' in df.columns:
        mm_mean = df['masfem_mturk'].mean(skipna=True)
        mm_std = df['masfem_mturk'].std(skipna=True)
        if pd.isna(mm_std) or mm_std == 0:
            df['masfem_mturk_z'] = df['masfem_mturk'] - mm_mean
        else:
            df['masfem_mturk_z'] = (df['masfem_mturk'] - mm_mean) / mm_std
    else:
        df['masfem_mturk_z'] = np.nan

    # Create a centered year variable to control for secular trends
    if 'year' in df.columns:
        df['year_c'] = df['year'] - df['year'].mean(skipna=True)
    else:
        df['year_c'] = np.nan

    # Ensure source is a string/categorical for later dummy creation
    if 'source' in df.columns:
        df['source'] = df['source'].astype('category')
    else:
        df['source'] = pd.Categorical([np.nan] * len(df))

    # Keep only the columns needed for modeling plus originals for transparency.
    # Do NOT drop rows here: let each model drop rows as appropriate so different analyses can use different subsets.
    keep_cols = list(df.columns)  # return all columns but with added derived variables

    return df

# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Runs the primary negative-binomial model predicting Deaths from masfem_z
    controlling for storm intensity and temporal/source covariates. Also runs
    a robustness OLS on logged damage (log_damage).

    Returns a dict with statsmodels result objects or wrappers: {'nb_results': <result wrapper>, 'ols_damage_results': <result wrapper or None>}.
    """
    results = {}

    # Helper to create a lightweight wrapper that exposes robust covariance & robust SEs
    class RobustResultWrapper:
        def __init__(self, res, cov_robust: np.ndarray):
            self._res = res
            self._cov_robust = cov_robust
            # keep params as attribute for convenience
            self.params = getattr(res, 'params', None)
            # compute robust bse
            self.bse = np.sqrt(np.diag(self._cov_robust)) if self._cov_robust is not None else None

        def summary(self):
            return self._res.summary()

        def cov_params(self):
            return self._cov_robust

        def get_robustcov_results(self, cov_type='HC3'):
            # emulate the interface expected by some code: return self
            return self

        def __getattr__(self, name):
            # delegate other attributes/methods to the underlying results object
            return getattr(self._res, name)

    def make_robust(res):
        """
        Try to obtain a robust-results object via res.get_robustcov_results.
        If not available, compute HC3 sandwich covariance and wrap the original result.
        """
        if res is None:
            return None
        # If the results object already exposes get_robustcov_results, call it
        if hasattr(res, 'get_robustcov_results'):
            try:
                robust = res.get_robustcov_results(cov_type='HC3')
                return robust
            except Exception:
                # fallback to manual sandwich cov
                pass
        # Compute HC3 robust covariance matrix and wrap
        try:
            cov_rob = cov_hc3(res)
        except Exception:
            cov_rob = None
        return RobustResultWrapper(res, cov_rob)

    # Prepare a working copy
    data = df.copy()

    # Define columns to use as controls (these must exist in the transformed dataframe)
    control_cols = ['wind', 'min', 'category', 'elapsedyrs', 'year_c']
    iv_col = 'masfem_z'
    dv_deaths = 'Deaths'
    dv_damage = 'log_damage'

    # ---- Model 1: Negative binomial for counts of deaths ----
    # Build design matrix: IV + controls + dummies for source
    model_cols = [iv_col] + control_cols + ['source']
    # Subset - keep rows where at least the columns exist
    sub = data[model_cols + [dv_deaths]].copy()

    # Create dummies for source (drop first to avoid collinearity). If source is all NaN, get_dummies will produce empty frame.
    if 'source' in sub.columns:
        source_dummies = pd.get_dummies(sub['source'], prefix='source', dummy_na=False, drop_first=True)
    else:
        source_dummies = pd.DataFrame(index=sub.index)

    X_nb = sub[[iv_col] + control_cols].join(source_dummies)
    y_nb = sub[dv_deaths]

    # Drop rows with missing values in X or y
    nb_df = pd.concat([X_nb, y_nb], axis=1).dropna()
    if nb_df.shape[0] < 10:
        raise ValueError('Too few rows available for the negative-binomial model after dropping NA.')

    y_nb_clean = nb_df[dv_deaths].astype(float)
    X_nb_clean = nb_df.drop(columns=[dv_deaths])
    X_nb_clean = sm.add_constant(X_nb_clean)

    # Fit a Negative Binomial GLM (handles overdispersion relative to Poisson)
    try:
        nb_model = sm.GLM(y_nb_clean, X_nb_clean, family=sm.families.NegativeBinomial())
        nb_res = nb_model.fit()
        nb_res_robust = make_robust(nb_res)
        results['nb_results'] = nb_res_robust
    except Exception as e:
        # If NegativeBinomial family fails, fall back to Poisson with robust SEs
        pois_model = sm.GLM(y_nb_clean, X_nb_clean, family=sm.families.Poisson())
        pois_res = pois_model.fit()
        pois_res_robust = make_robust(pois_res)
        results['nb_results'] = pois_res_robust
        results['nb_fallback_warning'] = f'NegativeBinomial failed with error: {e}. Used Poisson with robust SEs as fallback.'

    # ---- Model 2: Robust OLS on log damage (secondary check) ----
    # This model asks whether more feminine names are associated with lower precautionary behavior as proxied by higher logged damage
    dmg_cols = [iv_col] + control_cols + ['source']
    sub2 = data[dmg_cols + [dv_damage]].copy()
    if 'source' in sub2.columns:
        source_dummies2 = pd.get_dummies(sub2['source'], prefix='source', dummy_na=False, drop_first=True)
    else:
        source_dummies2 = pd.DataFrame(index=sub2.index)

    X_dmg = sub2[[iv_col] + control_cols].join(source_dummies2)
    y_dmg = sub2[dv_damage]

    dmg_df = pd.concat([X_dmg, y_dmg], axis=1).dropna()
    if dmg_df.shape[0] >= 10:
        y_dmg_clean = dmg_df[dv_damage].astype(float)
        X_dmg_clean = dmg_df.drop(columns=[dv_damage])
        X_dmg_clean = sm.add_constant(X_dmg_clean)
        ols_model = sm.OLS(y_dmg_clean, X_dmg_clean)
        ols_res = ols_model.fit()
        ols_res_robust = make_robust(ols_res)
        results['ols_damage_results'] = ols_res_robust
    else:
        results['ols_damage_results'] = None
        results['ols_warning'] = 'Too few rows to fit OLS on log_damage after dropping NA.'

    # Return the results (robust wrappers where possible)
    return results