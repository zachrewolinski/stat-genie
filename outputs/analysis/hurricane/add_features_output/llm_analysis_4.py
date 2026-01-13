from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/add_features_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into the cleaned dataframe used for modeling.

    Produces the following new columns (used in models):
      - alldeaths: integer count of deaths (copied from original but coerced to numeric)
      - log_alldeaths: log(alldeaths + 1)
      - log_ndam15: log(ndam15 + 1)
      - masfem_z: z-scored masfem (mean 0, sd 1)
      - masfem_mturk_z: z-scored masfem_mturk (alternative femininity measure)
      - gender_mf: binary indicator (0/1) copied from original
      - year_center: year - mean(year)

    Rows missing the primary IV (masfem) or DV (alldeaths) or core controls (wind, category, min) are dropped.
    """
    # Make a copy to avoid modifying caller's dataframe
    df = df.copy()

    # Ensure numeric columns are numeric
    numeric_cols = ['alldeaths', 'ndam15', 'masfem', 'masfem_mturk', 'gender_mf', 'wind', 'min', 'category', 'year']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the dependent variable or primary IV or key severity controls
    required_cols = ['alldeaths', 'masfem', 'wind', 'min', 'category', 'year']
    df = df.dropna(subset=required_cols)

    # Coerce alldeaths to integer counts (non-negative). Negative or missing -> drop
    df = df[df['alldeaths'] >= 0]
    df['alldeaths'] = df['alldeaths'].astype(int)

    # Log-transform for OLS robustness
    df['log_alldeaths'] = np.log(df['alldeaths'] + 1)

    # Damage variable: fill missing with 0 (no damage reported), then log-transform
    if 'ndam15' in df.columns:
        df['ndam15'] = df['ndam15'].fillna(0)
        df['log_ndam15'] = np.log(df['ndam15'] + 1)
    else:
        df['log_ndam15'] = 0.0

    # Standardize masfem and masfem_mturk for interpretability
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1.0)
    else:
        df['masfem_mturk_z'] = np.nan

    # Ensure gender_mf is 0/1 and no missing
    if 'gender_mf' in df.columns:
        df['gender_mf'] = df['gender_mf'].fillna(0).astype(int)
    else:
        df['gender_mf'] = 0

    # Center year to reduce collinearity and aid interpretation
    df['year_center'] = df['year'] - df['year'].mean()

    # Optionally create a high-category flag for descriptive checks
    df['high_category'] = (df['category'] >= 4).astype(int)

    # Keep only columns needed for modeling + a few originals for diagnostics
    keep_cols = [
        'alldeaths', 'log_alldeaths', 'masfem', 'masfem_z', 'masfem_mturk_z', 'gender_mf',
        'wind', 'min', 'category', 'log_ndam15', 'year', 'year_center', 'high_category'
    ]
    # Some columns may not exist; select intersection
    cols_present = [c for c in keep_cols if c in df.columns]
    df = df[cols_present]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit statistical models to test whether feminine hurricane names are associated with differences in fatalities.

    Primary specification: Negative Binomial regression predicting alldeaths (count) from standardized name femininity (masfem_z),
    controlling for storm intensity and damage.

    Robustness: OLS on log(alldeaths + 1) and an alternative NB model using the binary name-gender indicator (gender_mf).

    Returns a dictionary containing fitted model results objects.
    """
    import statsmodels.formula.api as smf

    results = {}

    # Ensure required columns exist
    required = ['alldeaths', 'masfem_z', 'wind', 'min', 'category', 'log_ndam15', 'year_center', 'gender_mf', 'log_alldeaths']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Formula for primary NB model (continuous femininity)
    formula_nb = 'alldeaths ~ masfem_z + wind + min + category + log_ndam15 + year_center'
    try:
        nb_model = smf.glm(formula=formula_nb, data=df, family=sm.families.NegativeBinomial()).fit()
        results['nb_masfem'] = nb_model
    except Exception as e:
        results['nb_masfem_error'] = str(e)

    # Alternative NB model using binary gender indicator
    formula_nb_bin = 'alldeaths ~ gender_mf + wind + min + category + log_ndam15 + year_center'
    try:
        nb_model_bin = smf.glm(formula=formula_nb_bin, data=df, family=sm.families.NegativeBinomial()).fit()
        results['nb_gender_mf'] = nb_model_bin
    except Exception as e:
        results['nb_gender_mf_error'] = str(e)

    # OLS robustness on log-deaths
    formula_ols = 'log_alldeaths ~ masfem_z + wind + min + category + log_ndam15 + year_center'
    try:
        ols_model = smf.ols(formula=formula_ols, data=df).fit()
        results['ols_masfem'] = ols_model
    except Exception as e:
        results['ols_masfem_error'] = str(e)

    # Additional sensitivity: substitute masfem_mturk_z if available
    if 'masfem_mturk_z' in df.columns and df['masfem_mturk_z'].notnull().any():
        formula_nb_mturk = 'alldeaths ~ masfem_mturk_z + wind + min + category + log_ndam15 + year_center'
        try:
            nb_model_mturk = smf.glm(formula=formula_nb_mturk, data=df, family=sm.families.NegativeBinomial()).fit()
            results['nb_masfem_mturk'] = nb_model_mturk
        except Exception as e:
            results['nb_masfem_mturk_error'] = str(e)

    # Return fitted results (statsmodels results objects) for downstream inspection
    return results


