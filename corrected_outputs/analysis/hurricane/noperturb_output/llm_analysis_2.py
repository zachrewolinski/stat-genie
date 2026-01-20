from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/hurricane/noperturb_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into the analysis-ready dataframe.

    Produces these columns used in modeling:
      - log_alldeaths: np.log(alldeaths + 1)
      - masfem_c: masfem mean-centered
      - gender_female: integer version of gender_mf (0/1)
      - log_ndam15: np.log(ndam15 + 1)
      - year_c: year mean-centered
    Also ensures numeric types for modeling and drops rows with missing values on the required vars.
    """
    df = df.copy()

    # Ensure columns exist
    required = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'ndam15', 'elapsedyrs', 'year']
    present = [c for c in required if c in df.columns]

    # Coerce numeric columns to numeric where present
    for c in present:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the key dependent or independent vars
    df = df.dropna(subset=['alldeaths', 'masfem'])

    # For stable multivariate models we need the main controls as well; drop rows missing them
    # If some controls are missing in the dataset, we don't drop on those
    controls_needed = [c for c in ['wind', 'min', 'category', 'ndam15', 'elapsedyrs', 'year'] if c in df.columns]
    if controls_needed:
        df = df.dropna(subset=controls_needed)

    # Derived columns
    # Log-transform deaths (many zeros, skewed): interpretable as percent-change-like effect
    df['log_alldeaths'] = np.log(df['alldeaths'].astype(float) + 1)

    # Centered masfem for interpretability
    df['masfem_c'] = df['masfem'].astype(float) - df['masfem'].astype(float).mean()

    # Binary female name indicator
    if 'gender_mf' in df.columns:
        df['gender_female'] = df['gender_mf'].astype(int)
    else:
        # if gender_mf is missing entirely, create NaNs so downstream code fails clearly
        df['gender_female'] = np.nan

    # Log of normalized damages (2015 USD) to control for exposure/scale
    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log(df['ndam15'].astype(float) + 1)

    # Mean-centered year to capture trend (alternative to elapsedyrs)
    if 'year' in df.columns:
        df['year_c'] = df['year'].astype(float) - df['year'].astype(float).mean()

    # Ensure category is integer (we will treat it as categorical in the model)
    if 'category' in df.columns:
        df['category'] = df['category'].astype(int)

    # Final safety drop: make sure model columns exist and have no missing
    model_cols = ['log_alldeaths', 'masfem_c', 'gender_female', 'wind', 'min', 'category', 'elapsedyrs']
    # include optional controls if present
    if 'log_ndam15' in df.columns:
        model_cols.append('log_ndam15')
    if 'year_c' in df.columns:
        model_cols.append('year_c')

    model_cols_present = [c for c in model_cols if c in df.columns]
    df = df.dropna(subset=model_cols_present)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear regression predicting log(alldeaths + 1) from name femininity and controls.

    Primary specification: OLS on log(deaths + 1) with robust standard errors.
    Formula:
      log_alldeaths ~ masfem_c + gender_female + wind + min + C(category) + elapsedyrs + log_ndam15 + year_c

    Returns a statsmodels results object with robust (HC3) SEs.
    """
    import statsmodels.formula.api as smf

    # Build formula. Only include optional controls if present in df
    formula_parts = ['masfem_c', 'gender_female', 'wind', 'min', 'C(category)', 'elapsedyrs']
    if 'log_ndam15' in df.columns:
        formula_parts.append('log_ndam15')
    if 'year_c' in df.columns:
        formula_parts.append('year_c')

    formula = 'log_alldeaths ~ ' + ' + '.join(formula_parts)

    # Fit OLS and obtain robust (HC3) standard errors
    model_ols = smf.ols(formula, data=df).fit()
    results = model_ols.get_robustcov_results(cov_type='HC3')

    return results


