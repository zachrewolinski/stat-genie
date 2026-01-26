from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/noperturb_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataset into a dataframe suitable for regression analysis.

    Produces these new columns (all used in modeling):
      - log_alldeaths: np.log1p(alldeaths)
      - masfem_z: standardized masfem
      - masfem_mturk_z: standardized masfem_mturk (if present)
      - ndam15_log: np.log1p(ndam15)
      - year_centered: year - mean(year)
      - category: categorical dtype
      - source: categorical dtype

    Rows with missing values on variables required for the main specification are dropped.
    """

    df = df.copy()

    # Ensure expected columns exist and coerce to numeric where appropriate
    if 'alldeaths' not in df.columns:
        raise ValueError("Input dataframe must contain 'alldeaths' column")

    # Coerce numeric columns where appropriate (exclude 'category' to avoid forcing categorical values to numeric)
    for col in ['alldeaths', 'masfem', 'masfem_mturk', 'wind', 'min', 'ndam15', 'year', 'gender_mf']:
        if col in df.columns:
            # try to coerce to numeric where meaningful
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure gender_mf is a plain numeric dtype (float) so downstream modeling/patsy won't encounter pandas nullable dtypes
    if 'gender_mf' in df.columns:
        df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce').astype(float)

    # Basic derived variables
    df['log_alldeaths'] = np.log1p(df['alldeaths'].astype(float))

    # Standardize masfem if present
    if 'masfem' in df.columns:
        masfem_mean = df['masfem'].mean(skipna=True)
        masfem_std = df['masfem'].std(skipna=True)
        # Avoid division by zero
        if pd.isna(masfem_mean) or pd.isna(masfem_std) or masfem_std == 0:
            df['masfem_z'] = np.nan
        else:
            df['masfem_z'] = (df['masfem'] - masfem_mean) / masfem_std
    else:
        df['masfem_z'] = np.nan

    # Standardize masfem_mturk when available (sensitivity)
    if 'masfem_mturk' in df.columns:
        mt_mean = df['masfem_mturk'].mean(skipna=True)
        mt_std = df['masfem_mturk'].std(skipna=True)
        if pd.isna(mt_mean) or pd.isna(mt_std) or mt_std == 0:
            df['masfem_mturk_z'] = np.nan
        else:
            df['masfem_mturk_z'] = (df['masfem_mturk'] - mt_mean) / mt_std
    else:
        df['masfem_mturk_z'] = np.nan

    # Log transform damages (2015 dollars)
    if 'ndam15' in df.columns:
        df['ndam15_log'] = np.log1p(df['ndam15'].astype(float))
    else:
        df['ndam15_log'] = np.nan

    # Center year
    if 'year' in df.columns:
        df['year_centered'] = df['year'].astype(float) - df['year'].astype(float).mean()
    else:
        df['year_centered'] = np.nan

    # Category and source as categorical
    if 'category' in df.columns:
        df['category'] = df['category'].astype('category')
    if 'source' in df.columns:
        df['source'] = df['source'].astype('category')

    # Select rows with required data for main specification
    required_cols = ['alldeaths', 'masfem_z', 'wind', 'min', 'category', 'ndam15_log', 'year_centered']
    present_required = [c for c in required_cols if c in df.columns]
    # Drop rows with NA in any of the present required columns
    df = df.dropna(subset=present_required)

    # Keep original identifiers if present
    # Return the transformed dataframe with all created columns
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit primary and sensitivity models to test whether more feminine hurricane names are associated
    with higher fatalities after controlling for physical severity and other covariates.

    Returns a dictionary with fitted model results (statsmodels results objects) and textual summaries.

    Models run:
      1) OLS on log-alldeaths using standardized masfem (primary continuous IV)
      2) OLS on log-alldeaths using binary gender_mf (sensitivity)
      3) Negative Binomial GLM on alldeaths using standardized masfem (count model / robustness)
      4) Sensitivity OLS replacing masfem_z with masfem_mturk_z if available

    All models include controls: wind, min, categorical category, ndam15_log, year_centered, and source.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    results = {}
    df = df.copy()

    # Ensure columns used in formulas are present
    base_controls = ' + wind + min + ndam15_log + year_centered + C(category) + C(source)'

    # Model 1: OLS with masfem_z
    if 'masfem_z' in df.columns and df['masfem_z'].notna().any():
        formula_ols_masfem = 'log_alldeaths ~ masfem_z' + base_controls
        ols_masfem = smf.ols(formula_ols_masfem, data=df).fit()
        results['ols_masfem'] = ols_masfem
        results['ols_masfem_summary'] = ols_masfem.summary().as_text()
    else:
        results['ols_masfem'] = None
        results['ols_masfem_summary'] = 'masfem_z not available or all NA; model not run.'

    # Model 2: OLS with binary gender_mf
    if 'gender_mf' in df.columns and df['gender_mf'].notna().any():
        # Cast gender_mf to numeric (0/1) as a plain numpy dtype to avoid pandas nullable dtypes
        df['gender_mf_num'] = pd.to_numeric(df['gender_mf'], errors='coerce').astype(float)
        formula_ols_gender = 'log_alldeaths ~ gender_mf_num' + base_controls
        ols_gender = smf.ols(formula_ols_gender, data=df).fit()
        results['ols_gender'] = ols_gender
        results['ols_gender_summary'] = ols_gender.summary().as_text()
    else:
        results['ols_gender'] = None
        results['ols_gender_summary'] = 'gender_mf not available or all NA; model not run.'

    # Model 3: Negative Binomial (count) with masfem_z
    # Negative Binomial can handle counts and overdispersion; use alldeaths as response
    if 'masfem_z' in df.columns and df['masfem_z'].notna().any():
        formula_nb = 'alldeaths ~ masfem_z' + base_controls
        try:
            nb_model = smf.glm(formula_nb, data=df, family=sm.families.NegativeBinomial()).fit()
            results['nb_masfem'] = nb_model
            results['nb_masfem_summary'] = nb_model.summary().as_text()
        except Exception as e:
            results['nb_masfem'] = None
            results['nb_masfem_summary'] = f'Negative Binomial failed: {e}'
    else:
        results['nb_masfem'] = None
        results['nb_masfem_summary'] = 'masfem_z not available or all NA; NB model not run.'

    # Model 4: Sensitivity OLS with masfem_mturk_z if available
    if 'masfem_mturk_z' in df.columns and df['masfem_mturk_z'].notna().any():
        formula_ols_mturk = 'log_alldeaths ~ masfem_mturk_z' + base_controls
        ols_mturk = smf.ols(formula_ols_mturk, data=df).fit()
        results['ols_mturk'] = ols_mturk
        results['ols_mturk_summary'] = ols_mturk.summary().as_text()
    else:
        results['ols_mturk'] = None
        results['ols_mturk_summary'] = 'masfem_mturk_z not available or all NA; sensitivity model not run.'

    # Return full results dict (models + summaries). Users can inspect coefficients, p-values, CIs, etc.
    return results