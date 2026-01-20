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
    Transform the raw hurricane dataset into a modeling dataframe.

    Produces the following columns (used in modeling):
      - MasFem: continuous masculinity-femininity score (from feature4)
      - Female: binary indicator for female name (from feature6)
      - MaxWind: maximum wind speed at landfall (from feature13)
      - MinPressure: minimum pressure at landfall (from feature5)
      - Category: Saffir-Simpson hurricane category (from feature7)
      - Year: year of occurrence (from feature2)
      - Source: data source (from feature11)
      - MTurkMasFem: MTurk-derived masfem index (from feature12)
      - Deaths: raw deaths (from feature8)
      - Damage_2015: normalized damage in 2015 dollars (from feature14)
      - log_deaths: np.log(Deaths + 1)
      - log_damage2015: np.log(Damage_2015 + 1)

    The function also coerces types and drops rows missing the main variables.
    """
    # Rename columns to meaningful names used in modeling
    df = df.copy()
    rename_map = {
        'feature1': 'ID',
        'feature2': 'Year',
        'feature3': 'Name',
        'feature4': 'MasFem',
        'feature5': 'MinPressure',
        'feature6': 'Female',
        'feature7': 'Category',
        'feature8': 'Deaths',
        'feature9': 'Damage_2013',
        'feature10': 'YearsSince',
        'feature11': 'Source',
        'feature12': 'MTurkMasFem',
        'feature13': 'MaxWind',
        'feature14': 'Damage_2015'
    }
    df = df.rename(columns=rename_map)

    # Coerce numeric columns to numeric types where appropriate
    numeric_cols = ['MasFem', 'MinPressure', 'Female', 'Category', 'Deaths', 'Damage_2015', 'MaxWind', 'Year', 'MTurkMasFem']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Create primary dependent variables: log transforms to reduce skew
    if 'Deaths' in df.columns:
        df['log_deaths'] = np.log1p(df['Deaths'])
    if 'Damage_2015' in df.columns:
        df['log_damage2015'] = np.log1p(df['Damage_2015'])

    # Basic filtering: drop rows missing the primary DV and primary IV(s)
    required = []
    if 'log_deaths' in df.columns:
        required.append('log_deaths')
    required += ['MasFem', 'Female', 'MaxWind', 'MinPressure', 'Category', 'Year']
    required = [c for c in required if c in df.columns]
    df = df.dropna(subset=required)

    # Optionally center Year to improve numerical stability
    if 'Year' in df.columns:
        df['Year_c'] = df['Year'] - df['Year'].mean()
        # Keep a simple 'Year' column too, but modeling will use Year_c
        df['Year'] = df['Year'].astype(int)

    # Standardize MasFem and MTurkMasFem for interpretability (z-scores)
    df['MasFem_z'] = (df['MasFem'] - df['MasFem'].mean()) / (df['MasFem'].std(ddof=0) if df['MasFem'].std(ddof=0) != 0 else 1)
    if 'MTurkMasFem' in df.columns:
        df['MTurkMasFem_z'] = (df['MTurkMasFem'] - df['MTurkMasFem'].mean()) / (df['MTurkMasFem'].std(ddof=0) if df['MTurkMasFem'].std(ddof=0) != 0 else 1)

    # Keep only columns that are necessary for modeling to make downstream code simpler
    keep_cols = [c for c in ['ID', 'Name', 'MasFem', 'MasFem_z', 'Female', 'MTurkMasFem', 'MTurkMasFem_z', 'MaxWind', 'MinPressure', 'Category', 'Year', 'Year_c', 'Source', 'Deaths', 'Damage_2015', 'log_deaths', 'log_damage2015'] if c in df.columns]
    df = df[keep_cols]

    # Ensure categorical columns are appropriate dtype
    if 'Category' in df.columns:
        # Some Category values may be floats; coerce to integer where possible
        try:
            df['Category'] = df['Category'].astype(int)
        except Exception:
            # leave as-is if coercion fails
            pass
    if 'Source' in df.columns:
        df['Source'] = df['Source'].astype(str)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit OLS models testing whether feminine hurricane names (MasFem / Female) predict fewer precautionary actions inferred via higher fatalities or damages.

    Returns a dictionary with two fitted robust models (if corresponding DVs are available):
      - 'deaths_model': OLS predicting log_deaths
      - 'damage_model': OLS predicting log_damage2015

    Models control for MaxWind, MinPressure, Category (categorical), and Year (centered).
    Robust (HC3) standard errors are returned.
    """
    import statsmodels.formula.api as smf

    results = {}

    # Base formula using the continuous femininity score and the binary female indicator
    # We include both MasFem and Female to allow checking whether the continuous rating or the binary label drives effects.
    # Use MasFem_z (standardized) for coefficient interpretability if available; otherwise use MasFem.
    iv_mas = 'MasFem_z' if 'MasFem_z' in df.columns else 'MasFem'
    iv_mturk = 'MTurkMasFem_z' if 'MTurkMasFem_z' in df.columns else 'MTurkMasFem' if 'MTurkMasFem' in df.columns else None

    base_terms = [iv_mas, 'Female', 'MaxWind', 'MinPressure', 'C(Category)', 'Year_c']
    if iv_mturk is not None:
        base_terms.append(iv_mturk)

    formula = ' + '.join([t for t in base_terms if t is not None])

    # Model 1: log_deaths
    if 'log_deaths' in df.columns:
        fmla_deaths = 'log_deaths ~ ' + formula
        mod = smf.ols(formula=fmla_deaths, data=df).fit()
        # Get robust covariance (HC3) for heteroskedasticity-robust SEs
        mod_robust = mod.get_robustcov_results(cov_type='HC3')
        results['deaths_model'] = mod_robust

    # Model 2: log_damage2015 (robustness check)
    if 'log_damage2015' in df.columns:
        fmla_damage = 'log_damage2015 ~ ' + formula
        mod2 = smf.ols(formula=fmla_damage, data=df).fit()
        mod2_robust = mod2.get_robustcov_results(cov_type='HC3')
        results['damage_model'] = mod2_robust

    # Return dict of fitted models with robust covariances. Each entry is a statsmodels RegressionResultsWrapper
    return results


