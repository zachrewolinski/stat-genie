from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original dataframe into a cleaned dataframe with variables used in the models.

    Expected original columns (based on dataset schema):
      - feature6: storm name (string)
      - feature12: binary name-gender indicator (0 male, 1 female)
      - feature9: expert masfem index (continuous)
      - feature11: MTurk masfem index (continuous)
      - feature13: total deaths (count)
      - feature7: maximum wind speed at landfall
      - feature4: Saffir-Simpson category
      - feature14: minimum pressure
      - feature8: property damage normalized to 2015
      - feature5: year
      - feature3: years elapsed since the hurricane
      - feature2: unique id

    Returns a dataframe with the exact column names used in the modeling stage.
    """
    df = df.copy()

    # Rename raw features to meaningful column names used in later modeling
    rename_map = {
        'feature6': 'StormName',
        'feature12': 'FemaleName',
        'feature9': 'MasFem_Expert',
        'feature11': 'MasFem_MTURK',
        'feature13': 'Deaths',
        'feature7': 'WindSpeed',
        'feature4': 'Category',
        'feature14': 'MinPressure',
        'feature8': 'Damage2015',
        'feature5': 'Year',
        'feature3': 'StormAge',
        'feature2': 'StormID'
    }
    df = df.rename(columns=rename_map)

    # Keep only needed columns if they exist
    needed = ['StormID', 'StormName', 'FemaleName', 'MasFem_Expert', 'MasFem_MTURK', 'Deaths',
              'WindSpeed', 'Category', 'MinPressure', 'Damage2015', 'Year', 'StormAge']
    # Some datasets might not include all masfem columns; handle safely
    for col in needed:
        if col not in df.columns:
            df[col] = pd.NA

    # Convert types
    # FemaleName should be numeric (0/1). Coerce to numeric and fill invalid as NA
    df['FemaleName'] = pd.to_numeric(df['FemaleName'], errors='coerce')

    # Ensure Deaths is numeric integer count
    df['Deaths'] = pd.to_numeric(df['Deaths'], errors='coerce').fillna(0).astype(int)

    # Construct FemininityScore by averaging the available masfem indices
    # If both exist, take mean; if only one exists, use that one
    df['MasFem_Expert'] = pd.to_numeric(df['MasFem_Expert'], errors='coerce')
    df['MasFem_MTURK'] = pd.to_numeric(df['MasFem_MTURK'], errors='coerce')
    df['FemininityScore'] = df[['MasFem_Expert', 'MasFem_MTURK']].mean(axis=1, skipna=True)

    # Create logged deaths outcome for OLS robustness
    df['LogDeaths'] = np.log1p(df['Deaths'].astype(float))

    # Numeric conversions for control variables
    df['WindSpeed'] = pd.to_numeric(df['WindSpeed'], errors='coerce')
    df['Category'] = pd.to_numeric(df['Category'], errors='coerce')
    df['MinPressure'] = pd.to_numeric(df['MinPressure'], errors='coerce')
    df['Damage2015'] = pd.to_numeric(df['Damage2015'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['StormAge'] = pd.to_numeric(df['StormAge'], errors='coerce')

    # Center Year for interpretability
    if df['Year'].notna().any():
        df['Year_c'] = df['Year'] - df['Year'].mean()
    else:
        df['Year_c'] = pd.NA

    # Drop rows with missing critical values required for main analysis
    # We need at least: Deaths, FemaleName OR FemininityScore, and core intensity controls
    required_for_model = ['Deaths', 'WindSpeed', 'Category', 'MinPressure']
    # Keep rows that have Deaths and at least one femininity measure and basic controls
    has_fem_measure = df['FemaleName'].notna() | df['FemininityScore'].notna()
    has_controls = df[required_for_model].notna().all(axis=1)
    df = df[has_fem_measure & has_controls].reset_index(drop=True)

    # Final check: cast FemaleName to int where possible
    df.loc[df['FemaleName'].notna(), 'FemaleName'] = df.loc[df['FemaleName'].notna(), 'FemaleName'].astype(int)

    # Return only columns that are used in modeling (plus StormName/ID for reference)
    final_cols = ['StormID', 'StormName', 'FemaleName', 'FemininityScore', 'Deaths', 'LogDeaths',
                  'WindSpeed', 'Category', 'MinPressure', 'Damage2015', 'Year', 'Year_c', 'StormAge']
    # Ensure final columns exist in df
    for col in final_cols:
        if col not in df.columns:
            df[col] = pd.NA

    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit statistical models to test whether more feminine hurricane names are associated with fatalities.

    Models estimated:
      1) Negative binomial GLM predicting Deaths with binary FemaleName + controls.
      2) Negative binomial GLM predicting Deaths with continuous FemininityScore + controls.
      3) OLS predicting LogDeaths with binary FemaleName + controls (robustness).
      4) OLS predicting LogDeaths with FemininityScore + controls (robustness).

    Returns a dict of model result objects.
    """
    results = {}
    df = df.copy()

    # Select control variables to include in all specifications
    controls = ['WindSpeed', 'Category', 'MinPressure', 'Damage2015', 'Year_c', 'StormAge']

    # Build design matrices safely (drop rows with NA in chosen predictors)
    import statsmodels.api as sm

    # Helper to prepare X and fit GLM Negative Binomial
    def fit_nb(endog_col, exog_cols, df_local):
        df_loc = df_local.dropna(subset=[endog_col] + exog_cols)
        if df_loc.shape[0] < 10:
            raise ValueError('Too few observations to fit model')
        y = df_loc[endog_col]
        X = df_loc[exog_cols]
        X = sm.add_constant(X, has_constant='add')
        # Use GLM with NegativeBinomial family (estimates dispersion)
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
        res = model.fit()
        return res

    # Helper to fit OLS on logged outcome
    def fit_ols(endog_col, exog_cols, df_local):
        df_loc = df_local.dropna(subset=[endog_col] + exog_cols)
        if df_loc.shape[0] < 10:
            raise ValueError('Too few observations to fit model')
        y = df_loc[endog_col]
        X = df_loc[exog_cols]
        X = sm.add_constant(X, has_constant='add')
        model = sm.OLS(y, X)
        res = model.fit(cov_type='HC3')  # robust SEs
        return res

    # Specification A: Binary FemaleName
    exogs_a = ['FemaleName'] + controls
    try:
        nb_a = fit_nb('Deaths', exogs_a, df)
        results['nb_female_binary'] = nb_a
    except Exception as e:
        results['nb_female_binary_error'] = str(e)

    try:
        ols_a = fit_ols('LogDeaths', exogs_a, df)
        results['ols_logdeaths_female_binary'] = ols_a
    except Exception as e:
        results['ols_logdeaths_female_binary_error'] = str(e)

    # Specification B: Continuous FemininityScore
    exogs_b = ['FemininityScore'] + controls
    try:
        nb_b = fit_nb('Deaths', exogs_b, df)
        results['nb_femscore'] = nb_b
    except Exception as e:
        results['nb_femscore_error'] = str(e)

    try:
        ols_b = fit_ols('LogDeaths', exogs_b, df)
        results['ols_logdeaths_femscore'] = ols_b
    except Exception as e:
        results['ols_logdeaths_femscore_error'] = str(e)

    # Return results dictionary. The caller can inspect .summary() for each fitted result.
    return results


