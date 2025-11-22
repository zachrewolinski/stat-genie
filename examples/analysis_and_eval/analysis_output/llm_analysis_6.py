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
    Prepare the hurricane dataset for modeling.
    - Drops rows missing core variables.
    - Converts relevant columns to numeric types.
    - Standardizes continuous covariates (z-scores) used in models.
    - Creates an interaction between name femininity and female-name indicator.
    - Creates a log-transformed damage variable for a secondary analysis (log_ndam15).
    - Encodes 'source' as a numeric categorical code (single column 'source_encoded') to keep control set compact.

    Returns the dataframe with additional columns:
      - masfem_z, wind_z, min_z, year_z, elapsedyrs_z
      - masfem_female_inter
      - log_ndam15
      - source_encoded
    """
    df = df.copy()

    # Ensure required columns exist and drop rows with missing critical values
    required = ['masfem', 'gender_mf', 'alldeaths', 'wind', 'category', 'min', 'ndam15', 'year', 'source', 'elapsedyrs']
    # convert to numeric where appropriate (errors -> NaN) then drop rows with NaNs in required
    for c in required:
        if c in ['masfem', 'gender_mf', 'alldeaths', 'wind', 'category', 'min', 'ndam15', 'year', 'elapsedyrs']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # 'source' keep as-is (categorical/string); if missing, will be dropped
    df = df.dropna(subset=required)

    # Standardize continuous predictors (z-scores). Use ddof=0 for population-like standardization (consistent across small samples).
    for col in ['masfem', 'wind', 'min', 'year', 'elapsedyrs']:
        zcol = f"{col}_z"
        # if constant, avoid division by 0
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            df[zcol] = 0.0
        else:
            df[zcol] = (df[col] - df[col].mean()) / std

    # Interaction term between standardized femininity and female-name indicator
    # Ensure gender_mf is numeric (0/1)
    df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce')
    df['masfem_female_inter'] = df['masfem_z'] * df['gender_mf']

    # Log-transform of damage (secondary DV) to reduce skewness
    df['log_ndam15'] = np.log1p(df['ndam15'])

    # Encode source as a single categorical code (keeps control set compact and column names stable)
    df['source_encoded'] = pd.Categorical(df['source']).codes

    # Return only the augmented dataframe (we keep all original columns plus the derived columns)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fits two models addressing the research question:
      1) Primary model: Negative binomial GLM predicting alldeaths (count) from name femininity and controls.
         - Rationale: fatalities are counts and over-dispersion is expected; NB GLM handles that.
      2) Secondary model: OLS predicting log-transformed property damage (log_ndam15) with robust SEs.

    Returns a dict with the fitted results objects: {'neg_binom': nb_results, 'ols_ndam': ols_results}
    """
    import statsmodels.api as sm

    # Define predictors used in both models (must match columns produced by transform)
    X_cols = [
        'masfem_z',           # main IV (standardized femininity)
        'gender_mf',          # binary female-name indicator
        'masfem_female_inter',# interaction (femininity x female name)
        'wind_z',             # standardized wind speed
        'min_z',              # standardized minimum pressure
        'category',           # category (1-5)
        'year_z',             # standardized year
        'elapsedyrs_z',       # standardized elapsed years
        'source_encoded'      # encoded data source
    ]

    # Ensure X columns exist in df
    missing = [c for c in X_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for model: {missing}")

    X = df[X_cols].astype(float)
    X = sm.add_constant(X)

    # Primary model: Negative Binomial for counts (alldeaths)
    y = df['alldeaths'].astype(float)
    # Fit a NB GLM (uses mean-variance relationship appropriate for over-dispersed counts)
    nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    nb_results = nb_model.fit()

    # Secondary model: OLS on log(ndam15 + 1) as a robustness check for property damage
    y2 = df['log_ndam15'].astype(float)
    ols_model = sm.OLS(y2, X)
    ols_results = ols_model.fit(cov_type='HC3')  # robust SEs

    # Return both fitted model results for inspection
    return {
        'neg_binom': nb_results,
        'ols_ndam': ols_results
    }


