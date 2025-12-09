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
    Transform original hurricane dataset into a modeling dataframe.

    Produces the following columns used later in the model:
      - StormName: original name (feature3)
      - MasFem: raw masculinity-femininity index (feature4)
      - MasFem_z: z-scored MasFem (standardized)
      - FemaleName: binary name-gender indicator (feature6) (0 male, 1 female)
      - Deaths: number of deaths (feature8) (int)
      - LogDeaths: log(Deaths + 1) (for sensitivity/OLS)
      - MaxWind: maximum wind speed at landfall (feature13)
      - MinPressure: minimum pressure at landfall (feature5)
      - Saffir: Saffir-Simpson category (feature7) (kept as numeric/categorical)
      - Damage2015: inflation/wealth/pop normalized damage (feature14)
      - LogDamage2015: log(Damage2015 + 1)
      - Year: year of storm (feature2)
      - Decade: categorical decade derived from Year
      - Source: data source (feature11)

    The function drops rows missing the primary variables required for the main analysis
    (MasFem and Deaths and core controls). It avoids in-place destructive changes by returning
    the transformed dataframe.
    """

    # Make a copy to avoid modifying original
    df = df.copy()

    # Map input columns (dataset schema) to clearer names
    # feature3: storm name, feature4: masfem index, feature6: female name indicator
    # feature8: deaths, feature14: damage adjusted to 2015
    # feature13: max wind, feature5: min pressure, feature7: Saffir category
    # feature2: year, feature11: source
    rename_map = {
        'feature3': 'StormName',
        'feature4': 'MasFem',
        'feature6': 'FemaleName',
        'feature8': 'Deaths',
        'feature14': 'Damage2015',
        'feature13': 'MaxWind',
        'feature5': 'MinPressure',
        'feature7': 'Saffir',
        'feature2': 'Year',
        'feature11': 'Source',
        'feature10': 'YearsSince'
    }
    df = df.rename(columns=rename_map)

    # Ensure expected renamed columns exist before further transformations to avoid KeyErrors
    expected_cols = list(rename_map.values())
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Ensure numeric types where expected
    for col in ['MasFem', 'FemaleName', 'Deaths', 'Damage2015', 'MaxWind', 'MinPressure', 'Saffir', 'Year']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing primary variables (MasFem or Deaths)
    # Both columns are guaranteed to exist (may be all-NaN); drop rows where either is missing.
    df = df.dropna(subset=['MasFem', 'Deaths'])

    # Make sure Deaths is integer (counts) and non-negative
    # Convert to int safely (after dropping NA), and remove negative counts if any
    df['Deaths'] = pd.to_numeric(df['Deaths'], errors='coerce').fillna(0).astype(int)
    df = df[df['Deaths'] >= 0]

    # Log-transform damage to reduce skew (add 1 to avoid log(0))
    if 'Damage2015' in df.columns:
        df['Damage2015'] = pd.to_numeric(df['Damage2015'], errors='coerce')
        df['Damage2015'] = df['Damage2015'].fillna(0)
        df['LogDamage2015'] = np.log(df['Damage2015'] + 1)
    else:
        df['Damage2015'] = 0.0
        df['LogDamage2015'] = 0.0

    # Standardize MasFem (z-score) for interpretability
    # Use population std (ddof=0) as in original code; guard against zero std
    masfem_mean = df['MasFem'].mean()
    masfem_std = df['MasFem'].std(ddof=0)
    if pd.isna(masfem_std) or masfem_std == 0:
        masfem_std = 1.0
    df['MasFem_z'] = (df['MasFem'] - masfem_mean) / masfem_std

    # Binary indicator FemaleName: ensure 0/1
    # Convert to numeric, coerce NA to 0, and clip to 0/1
    df['FemaleName'] = pd.to_numeric(df['FemaleName'], errors='coerce').fillna(0).astype(int)
    df['FemaleName'] = df['FemaleName'].clip(lower=0, upper=1)

    # Create a logged deaths variable for sensitivity OLS
    df['LogDeaths'] = np.log(df['Deaths'] + 1)

    # Ensure MaxWind and MinPressure numeric and fillna with median if missing (conservative)
    if 'MaxWind' in df.columns:
        df['MaxWind'] = pd.to_numeric(df['MaxWind'], errors='coerce')
        median_maxwind = df['MaxWind'].median()
        if pd.isna(median_maxwind):
            median_maxwind = 0.0
        df['MaxWind'] = df['MaxWind'].fillna(median_maxwind)
    else:
        df['MaxWind'] = 0.0

    if 'MinPressure' in df.columns:
        df['MinPressure'] = pd.to_numeric(df['MinPressure'], errors='coerce')
        median_minpres = df['MinPressure'].median()
        if pd.isna(median_minpres):
            median_minpres = 0.0
        df['MinPressure'] = df['MinPressure'].fillna(median_minpres)
    else:
        df['MinPressure'] = 0.0

    # Saffir as categorical: keep as-is but fill missing with 0 (unknown)
    if 'Saffir' in df.columns:
        df['Saffir'] = pd.to_numeric(df['Saffir'], errors='coerce')
        df['Saffir'] = df['Saffir'].fillna(0).astype(int)
    else:
        df['Saffir'] = 0

    # Year -> Decade categorical for temporal control
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        decade = np.floor(df['Year'] / 10) * 10
        # Replace NaN decades with 0 then string
        decade = decade.fillna(0).astype(int)
        df['Decade'] = decade.astype(str)
    else:
        df['Year'] = np.nan
        df['Decade'] = 'Unknown'

    # Source as categorical string
    if 'Source' in df.columns:
        df['Source'] = df['Source'].fillna('Unknown').astype(str)
    else:
        df['Source'] = 'Unknown'

    # Keep only columns needed for modeling to keep dataframe compact
    keep_cols = ['StormName', 'MasFem', 'MasFem_z', 'FemaleName', 'Deaths', 'LogDeaths', 'MaxWind', 'MinPressure', 'Saffir', 'Damage2015', 'LogDamage2015', 'Year', 'Decade', 'Source']
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[keep_cols]

    # Final drop of any rows that still miss the primary variables
    df = df.dropna(subset=['MasFem_z', 'Deaths'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit the primary statistical model linking name femininity to fatalities (proxy for precautionary behavior).

    Primary model: Negative Binomial GLM predicting Deaths (count) from standardized
    femininity index (MasFem_z), FemaleName indicator, and controls for storm severity,
    damage (log), Saffir category, decade, and data source. Use robust (HC3) standard errors.

    Sensitivity: OLS on LogDeaths with the same covariates.

    Returns a dictionary with fitted result objects: {'nb_results': nb_results, 'ols_results': ols_results}
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Drop rows with missing covariates used in model
    required_covs = ['MasFem_z', 'FemaleName', 'Deaths', 'MaxWind', 'MinPressure', 'LogDamage2015', 'Saffir', 'Decade', 'Source']
    model_df = df.dropna(subset=required_covs)

    # If there is no data to fit the model, return empty results rather than letting patsy error
    if model_df.shape[0] == 0:
        return {
            'nb_results': None,
            'ols_results': None,
            'model_dataframe': model_df
        }

    # Build a formula. C() wraps categorical variables.
    formula = 'Deaths ~ MasFem_z + FemaleName + MaxWind + MinPressure + LogDamage2015 + C(Saffir) + C(Decade) + C(Source)'

    # Fit Negative Binomial GLM (appropriate for overdispersed counts)
    nb_results = None
    try:
        nb_model = smf.glm(formula=formula, data=model_df, family=sm.families.NegativeBinomial()).fit()
        # Get robust (HC3) covariance if available
        nb_results = nb_model.get_robustcov_results(cov_type='HC3')
    except Exception:
        # If NB fails (numerical issues or design matrix problems), fall back to Poisson with robust SEs
        try:
            nb_model = smf.glm(formula=formula, data=model_df, family=sm.families.Poisson()).fit()
            nb_results = nb_model.get_robustcov_results(cov_type='HC3')
        except Exception:
            # If even Poisson fails, leave nb_results as None
            nb_results = None

    # Sensitivity analysis: OLS on log(deaths + 1)
    ols_formula = 'LogDeaths ~ MasFem_z + FemaleName + MaxWind + MinPressure + LogDamage2015 + C(Saffir) + C(Decade) + C(Source)'
    ols_results = None
    try:
        ols_model = smf.ols(formula=ols_formula, data=model_df).fit()
        # Attach robust covariances (HC3) if possible
        try:
            ols_results = ols_model.get_robustcov_results(cov_type='HC3')
        except Exception:
            # If robust cov can't be attached, keep the plain OLS results
            ols_results = ols_model
    except Exception:
        ols_results = None

    # Return models; callers can print summary() on each if not None
    return {
        'nb_results': nb_results,
        'ols_results': ols_results,
        'model_dataframe': model_df
    }