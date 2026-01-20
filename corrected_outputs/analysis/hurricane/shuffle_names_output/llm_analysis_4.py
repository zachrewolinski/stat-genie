from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/hurricane/shuffle_names_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Simonsohn hurricane dataframe into a modeling-ready dataframe.
    Creates and cleans the following columns used in the model:
      - Femininity: continuous masculinity-femininity rating (from column 'name')
      - IsFemaleName: binary female-name indicator (from 'elapsedyrs')
      - Deaths: raw deaths (from 'ndam15')
      - LogDeaths: log(Deaths + 1)
      - MaxWind: storm wind speed proxy (from 'min')
      - MinPressure: minimum central pressure (from 'gender_mf')
      - SaffirCategory: numeric storm category (from 'masfem')
      - Year: year of storm (from 'alldeaths')
      - Damage: normalized damage (from 'ind') and LogDamage
    The function coerces to numeric where appropriate, drops rows missing the primary variables,
    and returns the cleaned dataframe with these exact column names.
    """
    df = df.copy()

    # Coerce key columns to numeric where appropriate (some metadata descriptions are inconsistent)
    # 'name' in this dataset is the masculinity-femininity index (continuous)
    df['Femininity'] = pd.to_numeric(df.get('name'), errors='coerce')

    # Binary female name indicator: 'elapsedyrs' is 0/1 in this dataset metadata
    df['IsFemaleName'] = pd.to_numeric(df.get('elapsedyrs'), errors='coerce')

    # Deaths: use 'ndam15' as the total number of deaths
    df['Deaths'] = pd.to_numeric(df.get('ndam15'), errors='coerce')

    # Storm intensity controls
    # 'min' appears to be a wind-speed like variable in this schema (75-190). We'll keep it as MaxWind.
    df['MaxWind'] = pd.to_numeric(df.get('min'), errors='coerce')

    # 'gender_mf' in the provided schema appears to contain pressure-like values (909-1002). Keep as MinPressure.
    df['MinPressure'] = pd.to_numeric(df.get('gender_mf'), errors='coerce')

    # Saffir-Simpson category or analogous ordinal severity (from 'masfem')
    df['SaffirCategory'] = pd.to_numeric(df.get('masfem'), errors='coerce')

    # Year of the storm (metadata indicates 'alldeaths' holds the year)
    df['Year'] = pd.to_numeric(df.get('alldeaths'), errors='coerce')

    # Economic damage (use 'ind' which in metadata is normalized damage). Create log transform.
    df['Damage'] = pd.to_numeric(df.get('ind'), errors='coerce')

    # Drop rows missing the core independent or dependent variables
    df = df.dropna(subset=['Femininity', 'Deaths'])

    # Fill or drop other control NA values conservatively: keep rows with at least some control info, but
    # we'll drop rows missing the essential intensity controls to avoid bias from extreme missingness.
    df = df.dropna(subset=['MaxWind', 'MinPressure'], how='any')

    # Compute log transforms for count/damage outcomes
    df['LogDeaths'] = np.log(df['Deaths'] + 1)
    df['LogDamage'] = np.log(df['Damage'].fillna(0) + 1)

    # Standardize the femininity score for interpretability (mean 0, sd 1)
    df['Fem_z'] = (df['Femininity'] - df['Femininity'].mean()) / (df['Femininity'].std(ddof=0) if df['Femininity'].std(ddof=0) != 0 else 1.0)

    # Final selected columns (ensures exact column names used downstream exist)
    final_cols = ['Femininity', 'Fem_z', 'IsFemaleName', 'Deaths', 'LogDeaths', 'MaxWind', 'MinPressure', 'SaffirCategory', 'Year', 'Damage', 'LogDamage', 'ndam', 'ndam15']
    # Keep existing ones among final_cols (some datasets may not have e.g., 'ndam')
    available = [c for c in final_cols if c in df.columns]
    return df[available]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear regression testing the association between hurricane name femininity and fatalities.
    Primary specification: OLS with log fatalities as the outcome.
    Model: LogDeaths ~ Femininity + MaxWind + MinPressure + SaffirCategory + Year + LogDamage

    Returns the fitted statsmodels OLS results object.
    """
    # Make a copy to avoid side effects
    data = df.copy()

    # Ensure required columns exist
    required = ['LogDeaths', 'Femininity', 'MaxWind', 'MinPressure']
    missing = [c for c in required if c not in data.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Prepare predictors; include additional controls when available
    predictors = ['Femininity', 'MaxWind', 'MinPressure']
    # Optional controls if present in dataframe
    for c in ['SaffirCategory', 'Year', 'LogDamage', 'IsFemaleName']:
        if c in data.columns:
            predictors.append(c)

    X = data[predictors].astype(float)
    X = sm.add_constant(X)
    y = data['LogDeaths'].astype(float)

    model_res = sm.OLS(y, X, missing='drop').fit()

    # Return the fitted results object so the caller can inspect summary, params, etc.
    return model_res


