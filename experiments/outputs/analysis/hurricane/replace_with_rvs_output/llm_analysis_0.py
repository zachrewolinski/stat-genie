from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/replace_with_rvs_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and clean the hurricane dataset. Returns a dataframe with the exact columns used in the models.

    Steps:
    - Drop rows missing the key outcome or main predictors/controls
    - Standardize masfem into masfem_z
    - Create FemaleName from gender_mf (0/1)
    - Create year_centered and log_ndam15 (robustness outcome)
    - Ensure types are appropriate
    """
    df = df.copy()

    # Ensure columns exist
    needed = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'elapsedyrs', 'year', 'ndam15']
    missing = [c for c in needed if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows missing the primary outcome or core predictors/controls
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind', 'min', 'category', 'year'])

    # Ensure numeric types
    for c in ['alldeaths', 'masfem', 'wind', 'min', 'category', 'elapsedyrs', 'year']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Re-drop if conversion produced NaNs in required columns
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind', 'min', 'category', 'year'])

    # Standardize masfem (z-score). Use population std (ddof=0) for stability with small N
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / df['masfem'].std(ddof=0)

    # Binary female name indicator (ensure int 0/1)
    # If gender_mf is not exactly 0/1, coerce to binary by thresholding at 0.5
    df['FemaleName'] = df['gender_mf'].apply(lambda x: 1 if float(x) >= 0.5 else 0).astype(int)

    # Center year to control for secular trends while keeping intercept interpretable
    df['year_centered'] = df['year'] - df['year'].mean()

    # Robustness outcome: log-transformed adjusted damages (ndam15) + 1
    df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')
    df['log_ndam15'] = np.log(df['ndam15'].fillna(0) + 1)

    # Ensure alldeaths is integer-like (counts)
    df['alldeaths'] = df['alldeaths'].astype(int)

    # Keep only columns necessary for models (but return rest as well is harmless)
    # Return dataframe with the following columns guaranteed to exist for modeling:
    required_out_cols = ['alldeaths', 'masfem_z', 'FemaleName', 'wind', 'min', 'category', 'elapsedyrs', 'year_centered', 'log_ndam15']
    for c in required_out_cols:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' in the transformed dataframe but it's missing.")

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit the primary and robustness models to test whether more feminine hurricane names are associated
    with outcomes that would be expected if the public took fewer precautionary measures.

    Primary model: Negative binomial regression predicting alldeaths (count) from masfem_z
    and controls (wind, min, category, elapsedyrs, year_centered).

    Robustness model: OLS predicting log(ndam15 + 1) (adjusted damages) from the same predictors.

    Returns a dict with fitted statsmodels results objects.
    """
    df = df.copy()

    # Define predictors and add constant
    predictors = ['masfem_z', 'wind', 'min', 'category', 'elapsedyrs', 'year_centered']
    for p in predictors:
        if p not in df.columns:
            raise ValueError(f"Predictor column '{p}' not found in dataframe passed to model().")

    X = df[predictors]
    X = sm.add_constant(X)

    # Primary: negative binomial for count outcome (alldeaths)
    y_count = df['alldeaths']

    # Use GLM with NegativeBinomial family
    nb_model = sm.GLM(y_count, X, family=sm.families.NegativeBinomial()).fit()

    # Robustness: OLS on logged damages
    if 'log_ndam15' not in df.columns:
        raise ValueError("Column 'log_ndam15' required for robustness model not found in dataframe.")
    y_damage = df['log_ndam15']
    ols_damage_model = sm.OLS(y_damage, X).fit()

    # Also provide a simple alternative specification using FemaleName (binary) for transparency
    X_bin = df[['FemaleName', 'wind', 'min', 'category', 'elapsedyrs', 'year_centered']]
    X_bin = sm.add_constant(X_bin)
    nb_model_female = sm.GLM(df['alldeaths'], X_bin, family=sm.families.NegativeBinomial()).fit()

    results = {
        'nb_model_masfem_z': nb_model,
        'ols_damage_masfem_z': ols_damage_model,
        'nb_model_female_binary': nb_model_female
    }

    return results


