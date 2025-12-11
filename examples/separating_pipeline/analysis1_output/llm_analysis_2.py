from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Columns required for the main analysis
    required_cols = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'year', 'elapsedyrs', 'ndam15']

    # Drop rows missing any of the required variables
    df = df.dropna(subset=required_cols)

    # Ensure integer counts for fatalities
    # Some datasets may have floats; round to nearest int if necessary but keep as integers for NB model
    df['alldeaths'] = df['alldeaths'].astype(float).round().astype(int)

    # Standardize masfem (z-score) for easier interpretation
    masfem_std = df['masfem'].std(ddof=0)
    if np.isnan(masfem_std) or masfem_std == 0:
        masfem_std = 1.0
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / masfem_std

    # Center year to reduce collinearity with intercept
    df['year_c'] = df['year'] - df['year'].mean()

    # Log-transform damages (ndam15) to reduce skew and stabilize variance
    # Use log(1 + ndam15) to handle zeros
    df['log_ndam15'] = np.log1p(df['ndam15'].astype(float))

    # Ensure numeric types for control variables
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    df['min'] = pd.to_numeric(df['min'], errors='coerce')
    df['category'] = pd.to_numeric(df['category'], errors='coerce')
    df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')
    df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce')

    # After conversions, drop any rows that became NA in model variables
    df = df.dropna(subset=['masfem_z', 'alldeaths', 'gender_mf', 'wind', 'min', 'category', 'year_c', 'elapsedyrs', 'log_ndam15'])

    # Keep only columns that will be used in modeling plus identifiers to make output usable
    keep_cols = ['ind', 'year', 'name', 'masfem', 'masfem_z', 'gender_mf', 'wind', 'min', 'category', 'alldeaths', 'ndam15', 'log_ndam15', 'elapsedyrs', 'year_c']
    existing_keep = [c for c in keep_cols if c in df.columns]
    df = df[existing_keep]

    return df


def model(df: pd.DataFrame) -> Any:
    # Prepare design matrix
    # The independent variable of interest: masfem_z (standardized femininity index)
    # Controls: gender_mf, wind, min, category, year_c, elapsedyrs, log_ndam15
    X_cols = ['masfem_z', 'gender_mf', 'wind', 'min', 'category', 'year_c', 'elapsedyrs', 'log_ndam15']

    # Ensure all X columns exist
    missing = [c for c in X_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    X = df[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # Dependent variable
    y = df['alldeaths'].astype(int)

    # Fit a Negative Binomial GLM to account for overdispersion in count data
    # Request robust covariance (HC3) during fitting to avoid calling unavailable post-hoc methods
    model_glm = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    fit = model_glm.fit(cov_type='HC3')

    # Return the fitted results object (with HC3 robust covariance)
    return fit