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
    Transform the raw hurricane dataset into a dataframe ready for modeling.

    Produces the following columns (used in the model):
      - FemScore_z: standardized femininity score from column 'name'
      - FemaleName: binary indicator from 'elapsedyrs' (0 male, 1 female)
      - LogDeaths: log(ndam15 + 1)
      - LogDamage: log(ind + 1)
      - MaxWind: numeric wind speed from 'wind'
      - MinPressure: numeric pressure from 'min'
      - CategoryScale: numeric Saffir-Simpson scale from 'masfem'
      - Year: numeric year from 'alldeaths'

    Also retains original hurricane name as 'HurricaneName' for reference.
    """
    # Work on a copy to avoid modifying caller's data
    df = df.copy()

    # Coerce relevant columns to numeric where appropriate (suppressing errors to NaN)
    numeric_cols = {
        'ndam15': 'ndam15',
        'name': 'name',
        'elapsedyrs': 'elapsedyrs',
        'wind': 'wind',
        'min': 'min',
        'masfem': 'masfem',
        'ind': 'ind',
        'alldeaths': 'alldeaths'
    }
    for col in numeric_cols.values():
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Keep hurricane name text (if present)
    if 'ndam' in df.columns:
        df['HurricaneName'] = df['ndam'].astype(str)
    else:
        df['HurricaneName'] = df.index.astype(str)

    # Drop rows missing core variables needed for the analysis
    df = df.dropna(subset=['ndam15', 'name', 'wind', 'min', 'ind'])

    # Dependent variable: log-transformed deaths (add 1 to handle zeros)
    df['LogDeaths'] = np.log(df['ndam15'].astype(float) + 1.0)

    # Damage: log-transform normalized damage variable 'ind'
    # Some damage values may be zero; add 1 before log
    df['LogDamage'] = np.log(df['ind'].astype(float).clip(lower=0.0) + 1.0)

    # Independent variable: femininity score (column 'name' in dataset)
    df['FemScore'] = df['name'].astype(float)
    # Standardize (z-score) for interpretability
    # Use population std (ddof=0) to avoid small-sample ddof behavior; either is acceptable
    fem_mean = df['FemScore'].mean()
    fem_std = df['FemScore'].std(ddof=0)
    if fem_std == 0 or np.isnan(fem_std):
        df['FemScore_z'] = 0.0
    else:
        df['FemScore_z'] = (df['FemScore'] - fem_mean) / fem_std

    # Binary female name indicator
    # According to schema, 'elapsedyrs' is 0 for male, 1 for female; coerce to int
    df['FemaleName'] = df['elapsedyrs'].astype(int)

    # Meteorological controls
    df['MaxWind'] = df['wind'].astype(float)
    df['MinPressure'] = df['min'].astype(float)

    # Storm category / severity scale (masfem column described as Saffir-Simpson in schema)
    df['CategoryScale'] = df['masfem'].astype(float)

    # Year of event
    df['Year'] = df['alldeaths'].astype(int)

    # Optionally drop any remaining rows with NaNs in the modeling columns
    model_cols = ['LogDeaths', 'FemScore_z', 'FemaleName', 'LogDamage', 'MaxWind', 'MinPressure', 'CategoryScale', 'Year']
    df = df.dropna(subset=model_cols)

    # Return only the columns needed for modeling plus hurricane name for reference
    return df[[
        'HurricaneName',
        'LogDeaths',
        'FemScore',
        'FemScore_z',
        'FemaleName',
        'LogDamage',
        'MaxWind',
        'MinPressure',
        'CategoryScale',
        'Year',
        'ndam15',
        'ind'
    ]]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression predicting log fatalities (LogDeaths) from femininity of name
    and controls for storm severity and damage. Returns the fitted statsmodels results object.

    Model specification (primary):
      LogDeaths ~ FemScore_z + FemaleName + LogDamage + MaxWind + MinPressure + CategoryScale + Year

    Robust (heteroskedasticity-consistent) standard errors are used (HC3).
    """
    # Ensure we work on a copy
    df = df.copy()

    # Required columns for the model
    required = ['LogDeaths', 'FemScore_z', 'FemaleName', 'LogDamage', 'MaxWind', 'MinPressure', 'CategoryScale', 'Year']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe missing required columns for modeling: {missing}")

    # Drop rows with any NA in model variables
    df = df.dropna(subset=required)

    # Define dependent and independent variables
    y = df['LogDeaths'].astype(float)
    X = df[['FemScore_z', 'FemaleName', 'LogDamage', 'MaxWind', 'MinPressure', 'CategoryScale', 'Year']].astype(float)

    # Add constant for intercept
    X = sm.add_constant(X)

    # Fit OLS with robust standard errors (HC3)
    ols_model = sm.OLS(y, X)
    results = ols_model.fit(cov_type='HC3')

    # Return the fitted results object so caller can inspect params, summary, etc.
    return results


