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

    # Ensure numeric columns are numeric, coerce errors to NaN
    numeric_cols = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'category', 'min', 'year', 'ndam15']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the key variables needed for the primary analysis
    required = [c for c in ['alldeaths', 'masfem', 'wind', 'category', 'min', 'year'] if c in df.columns]
    df = df.dropna(subset=required)

    # Create the dependent variable: log deaths as proxy for precautionary failure
    df['LogDeaths'] = np.log(df['alldeaths'] + 1)

    # Center the masfem variable for clearer interpretation of the intercept
    df['masfem_c'] = df['masfem'] - df['masfem'].mean()

    # Optionally keep the binary gender indicator (alternative IV)
    if 'gender_mf' in df.columns:
        df['gender_mf'] = df['gender_mf'].astype(float)

    # Keep only columns needed for modeling and interpretation
    keep = ['LogDeaths', 'alldeaths', 'masfem', 'masfem_c', 'gender_mf', 'wind', 'category', 'min', 'year', 'ndam15', 'name']
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> float:
    # Make a local copy
    df = df.copy()

    # Build design matrix for OLS on log(deaths)
    # Primary model: LogDeaths ~ masfem_c + wind + category + min + year
    X_cols = [c for c in ['masfem_c', 'wind', 'category', 'min', 'year'] if c in df.columns]
    X = df[X_cols]
    X = sm.add_constant(X)
    y = df['LogDeaths']

    # Fit OLS (robust standard errors could be added; primary quantity returned is the masfem coefficient)
    model = sm.OLS(y, X, missing='drop').fit()

    # Return the estimated coefficient on the centered masfem variable (effect of a one-unit increase in femininity
    # on log fatalities, controlling for storm strength and year). If masfem_c not in model, return NaN.
    if 'masfem_c' in model.params.index:
        coef = float(model.params['masfem_c'])
    else:
        coef = float('nan')

    return coef


