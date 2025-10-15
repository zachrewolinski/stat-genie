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

    # Keep the relevant columns for the analysis
    required = ['alldeaths', 'masfem', 'wind', 'min', 'category', 'year']
    # Drop rows with missing values in any of the required columns
    df = df.dropna(subset=required)

    # Create a standardized (z-scored) version of the masculinity-femininity index
    # Higher values indicate more feminine names in the original 'masfem' column
    df['z_masfem'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1)

    # Center year to improve numerical stability and interpretability
    df['year_centered'] = df['year'] - df['year'].mean()

    # Ensure numeric types for controls
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    df['min'] = pd.to_numeric(df['min'], errors='coerce')
    df['category'] = pd.to_numeric(df['category'], errors='coerce')

    # After coercion, drop any rows that became NA
    df = df.dropna(subset=['wind', 'min', 'category'])

    # Keep only columns needed for modeling (but preserve others if desired)
    # Final dataframe will contain: alldeaths, z_masfem, wind, min, category, year_centered
    df = df.loc[:, list(set(['alldeaths', 'z_masfem', 'wind', 'min', 'category', 'year_centered']) & set(df.columns))]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> float:
    # The transform() function should already have been applied; expect columns:
    # 'alldeaths', 'z_masfem', 'wind', 'min', 'category', 'year_centered'

    # Prepare design matrix
    exog_cols = ['z_masfem', 'wind', 'min', 'category', 'year_centered']
    X = df[exog_cols].copy()
    # Add constant
    X = sm.add_constant(X, has_constant='add')
    y = df['alldeaths']

    # Fit a Negative Binomial GLM for count outcome (alldeaths) to allow overdispersion
    # If convergence issues arise, a Poisson with robust SE could be used instead.
    try:
        model_glm = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
    except Exception:
        # Fallback: Poisson with robust covariance (in case NB fails)
        model_glm = sm.GLM(y, X, family=sm.families.Poisson()).fit(cov_type='HC0')

    # Extract the coefficient for the key independent variable (z_masfem)
    coef = float(model_glm.params['z_masfem'])

    # Return the single-number summary: the estimated coefficient for (standardized) femininity.
    # Interpretation: change in the (link-scale) expected deaths associated with a 1 SD increase in name femininity,
    # conditional on controls. For the NB/Poisson canonical log link, this is the log-multiplicative effect.
    return coef


