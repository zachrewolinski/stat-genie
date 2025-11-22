from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/projects/binyu/hao_huang/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformations performed:
    - Drop rows missing the key outcome, IV, or core controls.
    - Ensure alldeaths is integer and non-negative.
    - Create log_ndam15 = log(ndam15 + 1) to compress heavy-tailed damage values.
    - Mean-center year to year_c (controls secular trends).
    - Z-score continuous predictors (masfem, wind, min, log_ndam15) to aid interpretation and numerical stability.
    - Return a dataframe containing exactly the columns used in the model.
    """
    df = df.copy()

    # Required raw columns
    required = ['alldeaths', 'masfem', 'wind', 'min', 'ndam15', 'year', 'category']
    # Drop rows missing required variables
    df = df.dropna(subset=required)

    # Ensure numeric types
    df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce')
    df['masfem'] = pd.to_numeric(df['masfem'], errors='coerce')
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    df['min'] = pd.to_numeric(df['min'], errors='coerce')
    df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['category'] = pd.to_numeric(df['category'], errors='coerce')

    # Drop any rows that became NA after coercion
    df = df.dropna(subset=required)

    # Ensure alldeaths are non-negative integers
    df = df[df['alldeaths'] >= 0]
    # In case floats exist, convert to integers (counts)
    df['alldeaths'] = df['alldeaths'].astype(int)

    # Create log-transformed damage variable (ndam15 may be heavy tailed)
    df['log_ndam15'] = np.log(df['ndam15'].clip(lower=0) + 1)

    # Mean-center year
    df['year_c'] = df['year'] - df['year'].mean()

    # Z-score the continuous predictors used in the model for interpretability
    for col in ['masfem', 'wind', 'min', 'log_ndam15']:
        zcol = f"{col}_z"
        # population std (ddof=0) to be deterministic
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # if constant, create zeros
            df[zcol] = 0.0
        else:
            df[zcol] = (df[col] - df[col].mean()) / std

    # Keep only columns necessary for modeling
    keep_cols = ['alldeaths', 'masfem_z', 'wind_z', 'min_z', 'log_ndam15_z', 'year_c', 'category']
    # If some of these columns are missing for any reason, raise an informative error
    missing = [c for c in keep_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required transformed columns: {missing}")

    return df[keep_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a count regression (Negative Binomial) predicting alldeaths from name femininity
    controlling for objective storm intensity and damage. Negative Binomial is chosen
    because alldeaths is a non-negative integer outcome with substantial overdispersion.

    Model specification (in matrix form):
      alldeaths ~ 1 + masfem_z + wind_z + min_z + log_ndam15_z + year_c + category

    Returns the fitted statsmodels results object.
    """
    df = df.copy()

    # Define predictors and outcome
    y = df['alldeaths']
    X = df[['masfem_z', 'wind_z', 'min_z', 'log_ndam15_z', 'year_c', 'category']]

    # Add intercept
    X = sm.add_constant(X)

    # Fit Negative Binomial using GLM
    model_glm = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    results = model_glm.fit()

    # Print summary for quick inspection (can be removed if silent return is preferred)
    print(results.summary())

    return results


