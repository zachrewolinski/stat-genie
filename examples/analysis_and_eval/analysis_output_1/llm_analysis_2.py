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
    Transform the raw hurricane dataframe into a modeling-ready dataframe.

    Outputs (columns that are created or guaranteed):
      - masfem_scaled : z-scored masfem (higher = more feminine name)
      - gender_mf     : original binary indicator (0 male, 1 female)
      - alldeaths     : original death counts (numeric)
      - log_alldeaths : log(alldeaths + 1)
      - wind, min, category, elapsedyrs, source : cleaned controls
      - year_centered : year centered at mean
    """
    import numpy as np

    # Make a copy to avoid modifying input in-place
    df = df.copy()

    # Ensure key numeric columns exist and coerce types where appropriate
    numeric_cols = ['masfem', 'gender_mf', 'alldeaths', 'wind', 'min', 'category', 'year', 'elapsedyrs']
    for col in numeric_cols:
        if col in df.columns:
            # coerce to numeric, convert errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure source exists and is string/categorical
    if 'source' in df.columns:
        df['source'] = df['source'].fillna('unknown').astype(str)
    else:
        df['source'] = 'unknown'

    # Drop rows missing the essential predictors/outcome or core severity controls
    required_for_model = ['masfem', 'alldeaths', 'wind', 'min', 'category', 'year']
    existing_required = [c for c in required_for_model if c in df.columns]
    if existing_required:
        df = df.dropna(subset=existing_required)

    # Standardize masfem for better interpretability
    if 'masfem' in df.columns:
        # Use sample standard deviation (ddof=0) consistent across environments
        mean_m = df['masfem'].mean()
        std_m = df['masfem'].std(ddof=0)
        if std_m == 0 or np.isnan(std_m):
            # avoid division by zero: create zeros
            df['masfem_scaled'] = 0.0
        else:
            df['masfem_scaled'] = (df['masfem'] - mean_m) / std_m
    else:
        df['masfem_scaled'] = np.nan

    # Keep the binary gender_mf indicator as-is (0/1). If missing, fill with 0/NaN handling left to model.
    if 'gender_mf' not in df.columns:
        df['gender_mf'] = np.nan

    # Outcome: deaths (count). Ensure non-negative integers; coerce negative or NaN to 0 if necessary (alternatively drop)
    df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce')
    # Keep rows with non-missing alldeaths already enforced above; but ensure no negatives
    df.loc[df['alldeaths'] < 0, 'alldeaths'] = 0
    # Create a logged version for OLS robustness checks
    df['log_alldeaths'] = np.log(df['alldeaths'] + 1)

    # Year centering to aid interpretation and reduce collinearity with intercept
    df['year_centered'] = df['year'] - df['year'].mean()

    # Category and source as categorical variables
    if 'category' in df.columns:
        df['category'] = df['category'].astype('category')
    df['source'] = df['source'].astype('category')

    # Final: ensure wind, min, elapsedyrs numeric
    for col in ['wind', 'min', 'elapsedyrs']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows still containing NA in the key modeling columns to produce a clean modeling dataframe
    model_cols = ['masfem_scaled', 'alldeaths', 'wind', 'min', 'category', 'year_centered', 'elapsedyrs', 'source']
    existent_model_cols = [c for c in model_cols if c in df.columns]
    df = df.dropna(subset=existent_model_cols)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a negative-binomial GLM predicting hurricane deaths from name femininity
    while controlling for storm severity and temporal/reporting confounds.

    Returns a dict with two fitted models for robustness:
      - 'nb_model': GLM NegativeBinomial fitted on raw death counts
      - 'ols_model': OLS fitted on log(alldeaths + 1) with robust (HC3) standard errors
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Required columns (assumes transform has been run):
    # masfem_scaled, gender_mf, alldeaths, log_alldeaths, wind, min, category, year_centered, elapsedyrs, source

    # Build formula: primary specification uses continuous masfem_scaled
    formula_base = 'alldeaths ~ masfem_scaled + gender_mf + wind + min + C(category) + year_centered + elapsedyrs + C(source)'

    # Fit negative binomial GLM for count outcome (alldeaths)
    try:
        nb_model = smf.glm(formula=formula_base, data=df, family=sm.families.NegativeBinomial()).fit()
    except Exception as e:
        # If NB fails (rare), fall back to Poisson with robust cov
        nb_model = smf.glm(formula=formula_base, data=df, family=sm.families.Poisson()).fit(cov_type='HC3')

    # Robust OLS on log deaths as a complementary specification
    ols_formula = 'log_alldeaths ~ masfem_scaled + gender_mf + wind + min + C(category) + year_centered + elapsedyrs + C(source)'
    ols_model = smf.ols(formula=ols_formula, data=df).fit(cov_type='HC3')

    # Return the fitted results objects so the analyst can inspect coefficients, CIs, diagnostics
    return {
        'nb_model': nb_model,
        'ols_model': ols_model
    }


