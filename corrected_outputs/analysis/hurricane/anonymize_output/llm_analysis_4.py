from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/hurricane/anonymize_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataset into the final dataframe used for modeling.

    - Rename columns to descriptive names.
    - Drop rows missing the key IV (feature4) or DV (feature8) or essential controls (feature13).
    - Create log-transformed fatalities (log_deaths) and log-transformed damage (log_damage).
    - Standardize the masfem score (masfem_z) to aid interpretation and numerical stability.
    - Create year-centered variable and decade categorical variable.
    - Ensure category is categorical and female_name is numeric (0/1).

    Returns a dataframe containing at least the columns listed in the conceptual variables.
    """
    # Work on a copy
    df = df.copy()

    # Rename raw columns to descriptive names used in modeling
    rename_map = {
        'feature1': 'storm_id',
        'feature2': 'year',
        'feature3': 'name',
        'feature4': 'masfem',            # continuous masculinity-femininity index
        'feature5': 'min_pressure',      # minimum pressure at landfall
        'feature6': 'female_name',       # binary name gender (0 male, 1 female)
        'feature7': 'category',          # Saffir-Simpson category
        'feature8': 'deaths',            # total number of deaths
        'feature9': 'damage_2013',       # damage normalized to 2013
        'feature10': 'years_since',      # number of years elapsed since hurricane
        'feature11': 'source',
        'feature12': 'mturk_masfem',     # alternate masfem ratings
        'feature13': 'max_wind',         # maximum wind speed at landfall
        'feature14': 'damage_2015'       # damage normalized to 2015
    }
    df.rename(columns=rename_map, inplace=True)

    # Keep rows that have the key variables required for the analysis
    required_cols = ['masfem', 'deaths', 'max_wind', 'damage_2015', 'year', 'category']
    df = df.dropna(subset=required_cols)

    # Ensure numeric types where appropriate
    numeric_cols = ['masfem', 'min_pressure', 'female_name', 'category', 'deaths', 'max_wind', 'damage_2015', 'year', 'years_since']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Replace or coerce any remaining missingness in numeric controls (drop rows with missing core controls)
    df = df.dropna(subset=['max_wind', 'deaths', 'masfem', 'damage_2015', 'year'])

    # Dependent variable: log(deaths + 1)
    df['log_deaths'] = np.log(df['deaths'].clip(lower=0) + 1)

    # Control: log damage (damage_2015 may contain zeros). Use +1 transform to keep zeros.
    df['damage_2015'] = df['damage_2015'].fillna(0)
    df['log_damage'] = np.log(df['damage_2015'].clip(lower=0) + 1)

    # Independent variable: standardize masfem for interpretability (z-score)
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1)

    # Ensure female_name is binary 0/1
    # If female_name already 0/1 keep it; otherwise coerce non-zero to 1
    df['female_name'] = df['female_name'].apply(lambda x: 1 if x == 1 else (0 if x == 0 else (1 if x in [True, 'female', 'F', 'f'] else 0)))

    # Year-centered control
    df['year_center'] = df['year'] - df['year'].mean()

    # Decade categorical variable (for flexible temporal controls)
    # e.g., 1950 -> 1950s, 1960 -> 1960s etc. Represent as string category
    df['decade'] = ((df['year'] // 10) * 10).astype(int).astype(str) + 's'

    # Treat category as categorical
    df['category'] = df['category'].astype('category')

    # Keep only columns needed for modeling plus some identifiers for downstream checks
    keep_cols = [
        'storm_id', 'name', 'year', 'decade', 'category', 'masfem', 'masfem_z', 'female_name',
        'deaths', 'log_deaths', 'damage_2015', 'log_damage', 'max_wind', 'min_pressure', 'year_center'
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df[keep_cols]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit OLS regressions testing whether more feminine hurricane names are associated with fewer precautionary measures,
    using fatalities (log_deaths) as the outcome and controlling for storm intensity and exposure.

    Two main specifications are fit:
      - model_masfem: continuous masfem (standardized) as the key predictor
      - model_female: binary female_name indicator as an alternative predictor

    Both models include controls: max_wind, min_pressure, log_damage, categorical storm category, and year_center.
    Heteroskedasticity-robust (HC3) standard errors are used.

    Returns a dict with fitted results objects.
    """
    import statsmodels.formula.api as smf

    results = {}

    # Specification 1: continuous femininity (standardized)
    formula1 = 'log_deaths ~ masfem_z + max_wind + min_pressure + log_damage + C(category) + year_center'
    model1 = smf.ols(formula1, data=df).fit(cov_type='HC3')
    results['model_masfem'] = model1

    # Specification 2: binary female name indicator
    formula2 = 'log_deaths ~ female_name + max_wind + min_pressure + log_damage + C(category) + year_center'
    model2 = smf.ols(formula2, data=df).fit(cov_type='HC3')
    results['model_female_name'] = model2

    # Additional diagnostics (optional): return number of observations used in each model and descriptive stats of key vars
    diagnostics = {
        'n_obs': int(df.shape[0]),
        'masfem_mean': float(df['masfem'].mean()) if 'masfem' in df.columns else None,
        'masfem_std': float(df['masfem'].std(ddof=0)) if 'masfem' in df.columns else None,
        'deaths_mean': float(df['deaths'].mean()) if 'deaths' in df.columns else None
    }
    results['diagnostics'] = diagnostics

    return results


