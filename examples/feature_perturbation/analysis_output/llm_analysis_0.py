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
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric; coerce errors to NaN
    numeric_map = {
        'feature13': 'deaths',         # total number of deaths
        'feature9': 'masfem',          # masculinity-femininity index (higher = more feminine)
        'feature12': 'female_name',    # binary gender indicator (0 male, 1 female)
        'feature7': 'max_wind',        # maximum wind speed at landfall
        'feature4': 'category',        # Saffir-Simpson category
        'feature14': 'min_pressure',   # minimum central pressure
        'feature8': 'damage_2015',     # property damage normalized to 2015 values
        'feature5': 'year'             # year of hurricane
    }

    for orig_col, new_col in numeric_map.items():
        if orig_col in df.columns:
            df[new_col] = pd.to_numeric(df[orig_col], errors='coerce')
        else:
            # keep new_col absent/NaN if original not available
            df[new_col] = np.nan

    # Keep only rows with non-missing values on the core variables needed for modeling
    required = ['deaths', 'masfem', 'female_name', 'max_wind', 'category', 'min_pressure', 'damage_2015', 'year']
    df = df.dropna(subset=required)

    # Ensure deaths are integers and non-negative
    df['deaths'] = df['deaths'].astype(int)
    df.loc[df['deaths'] < 0, 'deaths'] = 0

    # Create log outcome for robustness checks
    df['log_deaths'] = np.log1p(df['deaths'])

    # Standardize continuous predictors (z-scores) for interpretability in regressions
    cont_vars = {
        'masfem': 'masfem_z',
        'max_wind': 'wind_z',
        'min_pressure': 'pressure_z',
        'damage_2015': 'damage_z',
        'year': 'year_z'
    }
    for raw, zname in cont_vars.items():
        # If constant (zero variance) this will give NaN; that's acceptable and will be handled later
        df[zname] = (df[raw] - df[raw].mean()) / df[raw].std()

    # Make sure binary female_name is 0/1
    df['female_name'] = df['female_name'].apply(lambda x: 1 if x == 1 else 0)

    # Keep only the columns necessary for modeling to keep the final dataframe compact
    model_cols = ['deaths', 'log_deaths', 'masfem', 'masfem_z', 'female_name', 'max_wind', 'wind_z', 'category', 'min_pressure', 'pressure_z', 'damage_2015', 'damage_z', 'year', 'year_z']
    present_cols = [c for c in model_cols if c in df.columns]
    df = df[present_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    import statsmodels.api as sm

    results = {}

    # Prepare design matrix for Negative Binomial models
    # Model A: continuous masfem_z as main independent variable
    X_a_cols = ['masfem_z', 'wind_z', 'pressure_z', 'damage_z', 'year_z', 'category']
    X_a = df[X_a_cols].copy()
    X_a = sm.add_constant(X_a, has_constant='add')
    y = df['deaths']

    # Fit Negative Binomial (appropriate for overdispersed count data)
    try:
        nb_a = sm.GLM(y, X_a, family=sm.families.NegativeBinomial()).fit()
        results['nb_masfem'] = nb_a
    except Exception as e:
        results['nb_masfem_error'] = str(e)

    # Model B: binary female_name as main independent variable (alternative operationalization)
    X_b_cols = ['female_name', 'wind_z', 'pressure_z', 'damage_z', 'year_z', 'category']
    X_b = df[X_b_cols].copy()
    X_b = sm.add_constant(X_b, has_constant='add')

    try:
        nb_b = sm.GLM(y, X_b, family=sm.families.NegativeBinomial()).fit()
        results['nb_female_binary'] = nb_b
    except Exception as e:
        results['nb_female_binary_error'] = str(e)

    # Robustness: OLS on log-deaths
    X_c_cols = ['masfem_z', 'wind_z', 'pressure_z', 'damage_z', 'year_z', 'category']
    X_c = df[X_c_cols].copy()
    X_c = sm.add_constant(X_c, has_constant='add')
    y_log = df['log_deaths']

    try:
        ols_c = sm.OLS(y_log, X_c).fit()
        results['ols_log_deaths_masfem'] = ols_c
    except Exception as e:
        results['ols_log_deaths_masfem_error'] = str(e)

    # Return a dictionary of fitted model result objects (or error messages) so the caller can inspect summaries
    return results


