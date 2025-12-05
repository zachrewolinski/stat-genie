from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/campus/austin.zane/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into a modeling-ready dataframe.

    Produces the following new columns used in the models:
      - masfem_scaled: z-scored masfem (primary IV)
      - masfem_mturk_scaled: z-scored masfem_mturk (alternative IV)
      - log_alldeaths_plus1: log(alldeaths + 1) for OLS robustness
      - log_ndam15_plus1: log(ndam15 + 1) for OLS on damages
      - wind_scaled, min_scaled: standardized controls
    Also ensures categorical variables are filled and numeric types are consistent.
    """

    # Ensure we don't modify original reference
    df = df.copy()

    # Required columns for modeling
    required_cols = [
        'alldeaths', 'ndam15', 'masfem', 'wind', 'min', 'category',
        'elapsedyrs', 'gender_mf', 'source'
    ]

    # If masfem_mturk exists, we'll use it too; otherwise create NaNs
    if 'masfem_mturk' not in df.columns:
        df['masfem_mturk'] = np.nan

    # Drop rows that are missing the core outcome(s) or primary IV
    df = df.dropna(subset=['alldeaths', 'masfem'])

    # Fill missing controls with reasonable placeholders where appropriate
    # For source, keep a placeholder category
    df['source'] = df['source'].fillna('unknown').astype(str)

    # Ensure numeric types
    numeric_cols = ['alldeaths', 'ndam15', 'masfem', 'masfem_mturk', 'wind', 'min', 'category', 'elapsedyrs', 'gender_mf']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Some rows may have become NaN in required numeric fields; drop rows missing necessary numeric controls minimally
    # We keep rows missing optional fields like masfem_mturk or ndam15 (if using deaths model)
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind', 'min', 'category', 'elapsedyrs', 'gender_mf'])

    # Create logged outcomes for robustness
    df['log_alldeaths_plus1'] = np.log(df['alldeaths'].astype(float) + 1.0)
    # ndam15 may have missing values; create log when present
    if 'ndam15' in df.columns:
        df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')
        df['log_ndam15_plus1'] = np.log(df['ndam15'].fillna(0.0).astype(float) + 1.0)
    else:
        df['ndam15'] = np.nan
        df['log_ndam15_plus1'] = np.nan

    # Standardize continuous predictors (z-score) for easier coefficient interpretation
    def zscore(col: pd.Series) -> pd.Series:
        return (col - col.mean()) / (col.std(ddof=0) if col.std(ddof=0) != 0 else 1.0)

    df['masfem_scaled'] = zscore(df['masfem'].astype(float))
    df['masfem_mturk_scaled'] = zscore(df['masfem_mturk'].astype(float))
    df['wind_scaled'] = zscore(df['wind'].astype(float))
    df['min_scaled'] = zscore(df['min'].astype(float))

    # Ensure category is integer/ordinal
    df['category'] = pd.to_numeric(df['category'], errors='coerce').astype('Int64')

    # Ensure gender_mf is 0/1
    df['gender_mf'] = df['gender_mf'].astype(float).fillna(0).astype(int)

    # Keep only columns required for modeling and diagnostics but preserve other metadata
    model_cols = [
        'masfem_scaled', 'masfem_mturk_scaled', 'gender_mf',
        'alldeaths', 'log_alldeaths_plus1', 'ndam15', 'log_ndam15_plus1',
        'wind_scaled', 'min_scaled', 'category', 'elapsedyrs', 'source'
    ]

    # Some columns may not exist for all rows; ensure they're present
    for c in model_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Return dataframe ready for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run the statistical models to test whether more feminine hurricane names predict greater adverse outcomes
    (consistent with fewer precautions taken).

    Primary specification: Negative Binomial regression of raw fatalities (alldeaths) on masfem_scaled,
    controlling for physical storm severity and data source.

    Robustness/specification checks returned as well:
      - OLS on log(alldeaths + 1)
      - OLS on log(ndam15 + 1) for economic damage

    Returns a dictionary of fitted model results objects.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    results = {}

    # Drop rows with missing values on variables used in the primary model
    df_nb = df.dropna(subset=['alldeaths', 'masfem_scaled', 'wind_scaled', 'min_scaled', 'category', 'elapsedyrs', 'gender_mf', 'source']).copy()

    # Primary model: Negative Binomial on counts of deaths
    # Use categorical source with C(source) so different sources get separate intercept adjustments
    formula_nb = 'alldeaths ~ masfem_scaled + wind_scaled + min_scaled + category + elapsedyrs + gender_mf + C(source)'
    try:
        nb_model = smf.glm(formula=formula_nb, data=df_nb, family=sm.families.NegativeBinomial()).fit()
        results['nb_alldeaths'] = nb_model
    except Exception as e:
        results['nb_alldeaths_error'] = str(e)

    # Robustness 1: OLS on log(alldeaths + 1)
    df_ols_deaths = df.dropna(subset=['log_alldeaths_plus1', 'masfem_scaled', 'wind_scaled', 'min_scaled', 'category', 'elapsedyrs', 'gender_mf', 'source']).copy()
    formula_ols_deaths = 'log_alldeaths_plus1 ~ masfem_scaled + wind_scaled + min_scaled + category + elapsedyrs + gender_mf + C(source)'
    try:
        ols_deaths = smf.ols(formula=formula_ols_deaths, data=df_ols_deaths).fit()
        results['ols_log_alldeaths'] = ols_deaths
    except Exception as e:
        results['ols_log_alldeaths_error'] = str(e)

    # Robustness 2: OLS on log(ndam15 + 1) for economic damages
    df_ols_dam = df.dropna(subset=['log_ndam15_plus1', 'masfem_scaled', 'wind_scaled', 'min_scaled', 'category', 'elapsedyrs', 'gender_mf', 'source']).copy()
    formula_ols_dam = 'log_ndam15_plus1 ~ masfem_scaled + wind_scaled + min_scaled + category + elapsedyrs + gender_mf + C(source)'
    try:
        ols_dam = smf.ols(formula=formula_ols_dam, data=df_ols_dam).fit()
        results['ols_log_ndam15'] = ols_dam
    except Exception as e:
        results['ols_log_ndam15_error'] = str(e)

    # Additional robustness: replace primary IV with masfem_mturk_scaled (if available)
    df_nb_alt = df.dropna(subset=['alldeaths', 'masfem_mturk_scaled', 'wind_scaled', 'min_scaled', 'category', 'elapsedyrs', 'gender_mf', 'source']).copy()
    if df_nb_alt.shape[0] > 0:
        formula_nb_alt = 'alldeaths ~ masfem_mturk_scaled + wind_scaled + min_scaled + category + elapsedyrs + gender_mf + C(source)'
        try:
            nb_model_alt = smf.glm(formula=formula_nb_alt, data=df_nb_alt, family=sm.families.NegativeBinomial()).fit()
            results['nb_alldeaths_mturk_iv'] = nb_model_alt
        except Exception as e:
            results['nb_alldeaths_mturk_iv_error'] = str(e)
    else:
        results['nb_alldeaths_mturk_iv_error'] = 'No rows with masfem_mturk_scaled present.'

    return results


