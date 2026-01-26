from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/positive_leading_statement_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into the modeling dataframe.

    Produces the following new/ensured columns used in the modeling stage:
      - masfem : continuous femininity score (keeps original)
      - gender_mf : binary female name indicator (keeps original)
      - alldeaths : raw death counts (keeps original)
      - log_alldeaths : log(1 + alldeaths) to stabilize variance for OLS
      - wind, min, category, elapsedyrs, source, year : kept
      - year_c : centered year = year - mean(year)

    Drops rows with missing values in key columns required for modeling.
    """

    # Make a shallow copy to avoid modifying caller's frame
    df = df.copy()

    # Ensure columns exist and coerce types for numeric columns used as IV/DV/controls
    numeric_cols = ['masfem', 'gender_mf', 'alldeaths', 'wind', 'min', 'category', 'elapsedyrs', 'year', 'ndam15']
    for col in numeric_cols:
        if col in df.columns:
            # coerce to numeric, errors -> NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure source is string/categorical
    if 'source' in df.columns:
        df['source'] = df['source'].astype(str)

    # Drop rows missing the primary IV or DV or core controls
    required_for_model = [c for c in ['masfem', 'alldeaths', 'wind', 'min', 'category', 'year', 'elapsedyrs', 'source'] if c in df.columns]
    df = df.dropna(subset=required_for_model)

    # Create log-transformed death variable for OLS modeling
    # add 1 to handle zeros
    df['log_alldeaths'] = np.log1p(df['alldeaths'].astype(float))

    # Create log-transformed damages variable as a secondary outcome (if present)
    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log1p(df['ndam15'].astype(float))

    # Center year to aid interpretability / reduce collinearity
    if 'year' in df.columns:
        df['year_c'] = df['year'].astype(float) - df['year'].astype(float).mean()

    # Ensure masculinity-femininity measure is numeric and in a reasonable range
    if 'masfem' in df.columns:
        df['masfem'] = pd.to_numeric(df['masfem'], errors='coerce')

    # Ensure binary gender variable is numeric 0/1
    if 'gender_mf' in df.columns:
        df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce').fillna(0).astype(int)

    # Final drop of any rows that became NaN after transformations for columns we will use
    final_required = [c for c in ['masfem', 'gender_mf', 'alldeaths', 'log_alldeaths', 'wind', 'min', 'category', 'year_c', 'elapsedyrs', 'source'] if c in df.columns]
    df = df.dropna(subset=final_required)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run several models testing the relationship between hurricane name femininity and harms.

    Models included:
      1) OLS on log(1+alldeaths) with continuous masfem and controls.
      2) OLS on log(1+alldeaths) with binary gender_mf and controls (alternative IV).
      3) Negative binomial GLM on alldeaths (counts) with continuous masfem and controls.

    Returns a dictionary with fitted results objects for further inspection.
    """

    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    results = {}

    # Verify required columns exist
    needed = ['log_alldeaths', 'alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'year_c', 'elapsedyrs', 'source']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Common formula RHS (controls only)
    rhs = 'wind + min + category + year_c + elapsedyrs + C(source)'

    # 1) OLS with continuous masfem
    formula1 = 'log_alldeaths ~ masfem + ' + rhs
    ols_masfem = smf.ols(formula1, data=df).fit(cov_type='HC3')
    results['ols_masfem'] = ols_masfem

    # 2) OLS with binary gender indicator
    formula2 = 'log_alldeaths ~ gender_mf + ' + rhs
    ols_gender = smf.ols(formula2, data=df).fit(cov_type='HC3')
    results['ols_gender_mf'] = ols_gender

    # 3) Negative binomial GLM on counts (alldeaths)
    # Use the same RHS; GLM will internally handle categorical C(source) when using formula API
    formula_nb = 'alldeaths ~ masfem + ' + rhs
    try:
        nb_model = smf.glm(formula_nb, data=df, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
        results['nb_masfem'] = nb_model
    except Exception as e:
        # If GLM NegativeBinomial fails for numerical reasons, fallback to Poisson with robust SE and report warning
        poisson_model = smf.glm(formula_nb, data=df, family=sm.families.Poisson()).fit(cov_type='HC3')
        results['nb_masfem_fallback_poisson'] = poisson_model
        results['nb_error'] = str(e)

    # Also fit NB for gender_mf as alternative
    formula_nb2 = 'alldeaths ~ gender_mf + ' + rhs
    try:
        nb_model2 = smf.glm(formula_nb2, data=df, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
        results['nb_gender_mf'] = nb_model2
    except Exception:
        # skip if fails
        pass

    # Return the fitted results so caller can inspect .summary() etc.
    return results


