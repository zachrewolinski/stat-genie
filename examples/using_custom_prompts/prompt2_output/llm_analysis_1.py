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
    Prepare and clean the hurricane dataset for modeling.
    - Ensures numeric types for relevant columns
    - Drops rows missing essential variables
    - Creates log(1 + alldeaths) as the dependent variable
    - Standardizes the masfem variable (z-score) so the coefficient is interpretable
    - Creates log(1 + ndam15) as an auxiliary transformed column (not necessarily used in the primary model)
    Returns the dataframe with all columns needed for the model.
    """
    df = df.copy()

    # Ensure relevant columns exist and coerce to numeric where appropriate
    for col in ['masfem', 'alldeaths', 'wind', 'min', 'category', 'year', 'ndam15', 'gender_mf']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the core variables needed for main analysis
    df = df.dropna(subset=['masfem', 'alldeaths', 'wind', 'min', 'category', 'year'])

    # Dependent variable: log(1 + alldeaths) to reduce skew and handle zeros
    df['log_alldeaths'] = np.log1p(df['alldeaths'].astype(float))

    # Standardize masfem (z-score) so coefficient reflects change per SD of perceived femininity
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # Binary female name indicator from provided gender_mf (0 male, 1 female) if present
    if 'gender_mf' in df.columns:
        df['female_name'] = df['gender_mf'].astype(int)

    # Additional transformations/sensitivity variables
    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log1p(df['ndam15'].astype(float))

    # Keep only columns required for modelling to avoid accidental references later
    required_cols = [
        'masfem', 'masfem_z', 'female_name' if 'female_name' in df.columns else None,
        'alldeaths', 'log_alldeaths', 'wind', 'min', 'category', 'year', 'ndam15' if 'ndam15' in df.columns else None,
        'log_ndam15' if 'log_ndam15' in df.columns else None
    ]
    required_cols = [c for c in required_cols if c is not None]

    # Return the dataframe filtered to the required columns plus any others kept implicitly
    return df.loc[:, df.columns.intersection(required_cols)]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> float:
    """
    Fit a linear model predicting log(1 + alldeaths) from perceived femininity of the hurricane name
    plus key controls for storm intensity and year. Returns the estimated regression coefficient for masfem_z.

    Model: log_alldeaths ~ masfem_z + wind + min + category + year

    We use OLS on the log-transformed deaths. The returned single number is the estimated coefficient
    for masfem_z (interpretable as the change in log(1+deaths) per standard-deviation increase in masfem).
    """
    # Verify transformed columns exist
    required = ['log_alldeaths', 'masfem_z', 'wind', 'min', 'category', 'year']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Prepare design matrix
    X = df[['masfem_z', 'wind', 'min', 'category', 'year']].astype(float)
    X = sm.add_constant(X)
    y = df['log_alldeaths'].astype(float)

    # Fit OLS with heteroskedasticity-robust standard errors (for inference if desired)
    ols_model = sm.OLS(y, X).fit(cov_type='HC3')

    # Extract coefficient for masfem_z
    coef = float(ols_model.params['masfem_z'])

    # Optionally, one could also extract p-value: ols_model.pvalues['masfem_z']
    return coef


