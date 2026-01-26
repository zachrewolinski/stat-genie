from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/negative_leading_statement_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric
    num_cols = ['masfem', 'masfem_mturk', 'gender_mf', 'min', 'category', 'alldeaths', 'ndam', 'ndam15', 'elapsedyrs', 'wind', 'year']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Create dependent-variable transformations
    # Raw death counts (alldeaths) are used in a count model. Also create log transform for diagnostics/OLS.
    if 'alldeaths' in df.columns:
        df['log_alldeaths'] = np.log1p(df['alldeaths'])

    # Use adjusted damage (ndam15) as a continuous proxy for public-side impacts; log-transform to reduce skew.
    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log1p(df['ndam15'])

    # Main independent variables
    # Standardize masfem (continuous femininity rating) so coefficients are interpretable per-SD change.
    if 'masfem' in df.columns:
        masfem_mean = df['masfem'].mean()
        masfem_std = df['masfem'].std(ddof=0)
        # Avoid division by zero
        if pd.isna(masfem_std) or masfem_std == 0:
            df['masfem_std'] = (df['masfem'] - masfem_mean).astype(float)
        else:
            df['masfem_std'] = ((df['masfem'] - masfem_mean) / masfem_std).astype(float)

    # Binary gender indicator from original coding (0=male, 1=female)
    # Use standard numpy numeric dtype (float) so downstream statsmodels does not receive pandas nullable dtypes.
    if 'gender_mf' in df.columns:
        df['gender_f'] = pd.to_numeric(df['gender_mf'], errors='coerce').astype(float)

    # Center the year to improve interpretability and reduce collinearity with intercept
    if 'year' in df.columns:
        df['year_c'] = (df['year'] - df['year'].mean()).astype(float)

    # Ensure other control columns are standard numeric dtypes
    for c in ['wind', 'min', 'category', 'elapsedyrs', 'alldeaths', 'ndam15']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)

    # Return the transformed dataframe containing all original columns plus derived ones
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Runs two main models:
    1) Negative binomial GLM for raw death counts (alldeaths) testing whether name femininity predicts more deaths after controlling for severity.
    2) OLS for log(ndam15) (adjusted damage) testing the same question for damages.

    Returns a dictionary with fitted model objects and human-readable summaries.
    """
    results = {}

    # --- Model 1: Death counts (Negative Binomial) ---
    # Select variables required for the death model
    required_death = ['alldeaths', 'masfem_std', 'wind', 'min', 'category', 'year_c', 'elapsedyrs']
    miss = [c for c in required_death if c not in df.columns]
    if len(miss) > 0:
        raise ValueError('Missing columns required for death model: ' + ','.join(miss))

    df_death = df[required_death].dropna()

    # Endog: counts (non-negative integers). Exog: intercept + controls
    endog = pd.to_numeric(df_death['alldeaths'], errors='coerce').astype(float)
    exog = df_death[['masfem_std', 'wind', 'min', 'category', 'year_c', 'elapsedyrs']].apply(pd.to_numeric, errors='coerce').astype(float)
    exog = sm.add_constant(exog, has_constant='add')

    # Fit a Negative Binomial GLM (robust SE). NB addresses overdispersion common in count data.
    try:
        nb_model = sm.GLM(endog, exog, family=sm.families.NegativeBinomial())
        nb_res = nb_model.fit(cov_type='HC3')
    except Exception as e:
        # Fallback: poisson with robust SE if NB fails
        nb_model = sm.GLM(endog, exog, family=sm.families.Poisson())
        nb_res = nb_model.fit(cov_type='HC3')
        results['death_model_warning'] = f'NegativeBinomial failed, used Poisson as fallback. Original error: {e}'

    results['death_model'] = nb_res
    results['death_summary'] = nb_res.summary().as_text()

    # Also report the coefficient and p-value for masfem_std specifically
    if 'masfem_std' in exog.columns:
        coef = nb_res.params.get('masfem_std', np.nan)
        pval = nb_res.pvalues.get('masfem_std', np.nan)
        results['death_masfem_coef'] = float(coef) if not pd.isna(coef) else None
        results['death_masfem_pvalue'] = float(pval) if not pd.isna(pval) else None

    # --- Model 2: Adjusted damage (log-ndam15) using OLS ---
    required_dam = ['log_ndam15', 'masfem_std', 'wind', 'min', 'category', 'year_c', 'elapsedyrs']
    miss2 = [c for c in required_dam if c not in df.columns]
    if len(miss2) > 0:
        raise ValueError('Missing columns required for damage model: ' + ','.join(miss2))

    df_dam = df[required_dam].dropna()
    y = pd.to_numeric(df_dam['log_ndam15'], errors='coerce').astype(float)
    X = df_dam[['masfem_std', 'wind', 'min', 'category', 'year_c', 'elapsedyrs']].apply(pd.to_numeric, errors='coerce').astype(float)
    X = sm.add_constant(X, has_constant='add')

    ols_model = sm.OLS(y, X)
    ols_res = ols_model.fit(cov_type='HC3')

    results['damage_model'] = ols_res
    results['damage_summary'] = ols_res.summary().as_text()

    if 'masfem_std' in X.columns:
        coef2 = ols_res.params.get('masfem_std', np.nan)
        pval2 = ols_res.pvalues.get('masfem_std', np.nan)
        results['damage_masfem_coef'] = float(coef2) if not pd.isna(coef2) else None
        results['damage_masfem_pvalue'] = float(pval2) if not pd.isna(pval2) else None

    # --- Robustness checks returned (optional) ---
    # 1) Binary gender indicator instead of continuous masfem
    if 'gender_f' in df.columns:
        req_bin = ['alldeaths', 'gender_f', 'wind', 'min', 'category', 'year_c', 'elapsedyrs']
        df_bin = df[req_bin].dropna()
        if len(df_bin) >= 10:
            endog_b = pd.to_numeric(df_bin['alldeaths'], errors='coerce').astype(float)
            exog_b = df_bin[['gender_f', 'wind', 'min', 'category', 'year_c', 'elapsedyrs']].apply(pd.to_numeric, errors='coerce').astype(float)
            exog_b = sm.add_constant(exog_b, has_constant='add')
            try:
                nb_b = sm.GLM(endog_b, exog_b, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
            except Exception:
                nb_b = sm.GLM(endog_b, exog_b, family=sm.families.Poisson()).fit(cov_type='HC3')
            results['death_model_gender_bin_summary'] = nb_b.summary().as_text()
            coef_g = nb_b.params.get('gender_f', np.nan)
            pval_g = nb_b.pvalues.get('gender_f', np.nan)
            results['death_gender_bin_coef'] = float(coef_g) if not pd.isna(coef_g) else None
            results['death_gender_bin_pvalue'] = float(pval_g) if not pd.isna(pval_g) else None

    # Return the results dictionary (includes model objects and text summaries)
    return results