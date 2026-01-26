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

    # Standardize/clean source into a small set of categories and create dummies
    # Normalize source string and map
    def _map_source(x):
        if pd.isna(x):
            return 'other'
        s = str(x).lower()
        if 'uri' in s:
            return 'uri'
        if 'wiki' in s:
            return 'wiki'
        if 'mwr' in s:
            return 'mwr'
        return 'other'

    df['source_simple'] = df['source'].apply(_map_source)
    # Create dummies, drop the reference 'uri' to avoid multicollinearity
    source_dummies = pd.get_dummies(df['source_simple'], prefix='source')
    if 'source_uri' in source_dummies.columns:
        source_dummies = source_dummies.drop(columns=['source_uri'])
    # Ensure consistent dummy columns exist (even if some categories absent)
    for col in ['source_mwr', 'source_wiki', 'source_other']:
        if col not in source_dummies.columns:
            source_dummies[col] = 0
    df = pd.concat([df, source_dummies[['source_mwr', 'source_wiki', 'source_other']]], axis=1)

    # Impute masfem_mturk with masfem mean when missing (so we can compare both IVs)
    if 'masfem_mturk' in df.columns:
        mturk_mean = df['masfem_mturk'].mean()
        df['masfem_mturk'] = df['masfem_mturk'].fillna(mturk_mean)

    # Drop rows missing key variables required for analysis
    required = ['masfem', 'gender_mf', 'alldeaths', 'ndam15', 'wind', 'min', 'category', 'year', 'elapsedyrs']
    # keep only those that exist in df
    required = [c for c in required if c in df.columns]
    df = df.dropna(subset=required)

    # Create log-transformed outcomes (log1p to handle zeros)
    df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce')
    df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')
    df['log_alldeaths'] = np.log1p(df['alldeaths'].clip(lower=0))
    df['log_ndam15'] = np.log1p(df['ndam15'].clip(lower=0))

    # Center continuous predictors to improve interpretability
    def center(col):
        return df[col] - df[col].mean()

    df['masfem_c'] = center('masfem')
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_c'] = center('masfem_mturk')
    else:
        df['masfem_mturk_c'] = df['masfem_c'].copy()

    df['wind_c'] = center('wind')
    df['min_c'] = center('min')
    df['year_c'] = center('year')
    df['elapsedyrs_c'] = center('elapsedyrs')

    # Make sure category is numeric (ordinal)
    df['category'] = pd.to_numeric(df['category'], errors='coerce')

    # Ensure gender_mf is numeric 0/1
    df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce').fillna(0).astype(int)

    # Keep only the columns needed for modeling to avoid passing extra junk
    cols_to_keep = [
        'masfem_c', 'masfem_mturk_c', 'gender_mf',
        'alldeaths', 'log_alldeaths', 'ndam15', 'log_ndam15',
        'wind_c', 'min_c', 'category', 'year_c', 'elapsedyrs_c',
        'source_mwr', 'source_wiki', 'source_other'
    ]
    existing = [c for c in cols_to_keep if c in df.columns]
    df = df[existing].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Runs a set of models testing whether feminine hurricane names predict smaller precautionary outcomes
    (proxied by deaths and damages) after controlling for storm severity and other covariates.

    Returns a dictionary with model objects and summaries.
    """
    results = {}

    # Prepare design matrix common to models
    exog_vars = ['masfem_c', 'wind_c', 'min_c', 'category', 'year_c', 'elapsedyrs_c', 'source_mwr', 'source_wiki', 'source_other', 'gender_mf']
    exog = df[exog_vars].astype(float).copy()
    exog = sm.add_constant(exog)

    # 1) OLS on log(total deaths) with HC3 robust SEs
    y_deaths = df['log_alldeaths']
    ols_deaths = sm.OLS(y_deaths, exog).fit(cov_type='HC3')
    results['ols_log_alldeaths'] = ols_deaths

    # 2) OLS on log(damages) with HC3 robust SEs
    y_dam = df['log_ndam15']
    ols_dam = sm.OLS(y_dam, exog).fit(cov_type='HC3')
    results['ols_log_ndam15'] = ols_dam

    # 3) Negative binomial on raw alldeaths (counts). Use GLM NegativeBinomial with log link.
    #    Add a tiny constant to counts if zeros are all zeros (GLM handles zeros fine). Keep same exog.
    y_counts = df['alldeaths'].fillna(0).astype(float)
    try:
        nb_model = sm.GLM(y_counts, exog, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
        results['nb_alldeaths'] = nb_model
    except Exception as e:
        results['nb_alldeaths_error'] = str(e)

    # 4) Robustness: replace masfem_c with masfem_mturk_c to see if MTurk ratings change inference
    exog_rb = exog.copy()
    if 'masfem_mturk_c' in df.columns:
        exog_rb['masfem_c'] = df['masfem_mturk_c']
        ols_deaths_rb = sm.OLS(df['log_alldeaths'], exog_rb).fit(cov_type='HC3')
        ols_dam_rb = sm.OLS(df['log_ndam15'], exog_rb).fit(cov_type='HC3')
        results['ols_log_alldeaths_mturkIV'] = ols_deaths_rb
        results['ols_log_ndam15_mturkIV'] = ols_dam_rb

    # 5) Additional simple tabulation: correlation between masfem and severity measures (wind, category)
    corr = {
        'corr_masfem_wind': float(np.corrcoef(df['masfem_c'], df['wind_c'])[0,1]),
        'corr_masfem_min': float(np.corrcoef(df['masfem_c'], df['min_c'])[0,1]),
        'corr_masfem_category': float(np.corrcoef(df['masfem_c'], df['category'])[0,1])
    }
    results['simple_correlations'] = corr

    # Summarize key coefficient (masfem_c) from OLS deaths and damages with coefficient, se, t, p, 95% CI
    def summarize_coef(model_obj, var='masfem_c'):
        if not hasattr(model_obj, 'params'):
            return None
        params = model_obj.params
        b = float(params.get(var, np.nan))
        se = float(model_obj.bse.get(var, np.nan))
        t = float(model_obj.tvalues.get(var, np.nan)) if hasattr(model_obj, 'tvalues') else None
        p = float(model_obj.pvalues.get(var, np.nan)) if hasattr(model_obj, 'pvalues') else None
        ci_low, ci_high = (None, None)
        try:
            ci = model_obj.conf_int().loc[var]
            ci_low, ci_high = float(ci[0]), float(ci[1])
        except Exception:
            pass
        return {'coef': b, 'se': se, 't': t, 'p': p, 'ci_lower': ci_low, 'ci_upper': ci_high}

    results['masfem_coef_ols_log_alldeaths'] = summarize_coef(ols_deaths, 'masfem_c')
    results['masfem_coef_ols_log_ndam15'] = summarize_coef(ols_dam, 'masfem_c')

    # Also include the same summary for MTurk IV if present
    if 'ols_log_alldeaths_mturkIV' in results:
        results['masfem_mturk_coef_ols_log_alldeaths'] = summarize_coef(results['ols_log_alldeaths_mturkIV'], 'masfem_c')
        results['masfem_mturk_coef_ols_log_ndam15'] = summarize_coef(results['ols_log_ndam15_mturkIV'], 'masfem_c')

    # Return results dictionary (models are statsmodels results objects; also include human-readable summaries)
    return results


