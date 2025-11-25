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
    Transform the raw hurricane dataframe into the final dataframe used for modeling.

    Produces the following new columns (exact names used in the model):
      - log_alldeaths : np.log1p(alldeaths)
      - log_ndam15   : np.log1p(ndam15)
      - masfem_std   : standardized masfem (z-score)
      - source_code  : integer code for the categorical 'source' column

    Also drops rows missing the key variables required for the analysis.
    """
    df = df.copy()

    # Required raw columns
    required_cols = [
        'alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category',
        'ndam15', 'year', 'elapsedyrs', 'source'
    ]

    # Drop rows missing any required column
    df = df.dropna(subset=required_cols)

    # Ensure numeric columns are numeric
    numeric_cols = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'ndam15', 'year', 'elapsedyrs']
    for c in numeric_cols:
        # coerce to numeric; invalid parsing to NaN will be dropped
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=numeric_cols)

    # Dependent variable: log-transform fatalities to reduce skew
    df['log_alldeaths'] = np.log1p(df['alldeaths'].astype(float))

    # Control: log-transformed damage (2015-normalized) as a proxy for exposure/economic impact
    df['log_ndam15'] = np.log1p(df['ndam15'].astype(float))

    # Independent variable: standardize masfem (z-score) to aid interpretation
    df['masfem_std'] = (df['masfem'].astype(float) - df['masfem'].astype(float).mean()) / (df['masfem'].astype(float).std(ddof=0) if df['masfem'].astype(float).std(ddof=0) != 0 else 1.0)

    # Ensure gender_mf is numeric 0/1
    df['gender_mf'] = df['gender_mf'].astype(int)

    # Encode source as an integer code to control for reporting/source differences
    # Factorize returns (codes, uniques); codes are 0..k-1
    df['source_code'] = pd.factorize(df['source'])[0]

    # Keep only columns relevant to the model plus original identifiers (if present)
    # Final dataframe must include all columns listed in the conceptual variables
    keep_cols = list(df.columns)  # keep everything for flexibility

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit OLS models predicting log fatalities from name femininity while controlling for physical severity
    and temporal/source variables. Returns a dict of fitted statsmodels results objects.

    Three models are estimated:
      1) model_masfem_cont: masfem_std (continuous) + controls
      2) model_gender_bin: gender_mf (binary female indicator) + controls
      3) model_both: both masfem_std and gender_mf + controls

    Robust (HC3) standard errors are used to mitigate heteroskedasticity.
    """
    df = df.copy()

    # Check that required transformed columns are present
    required = ['log_alldeaths', 'masfem_std', 'gender_mf', 'wind', 'min', 'category', 'year', 'elapsedyrs', 'log_ndam15', 'source_code']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"The following required columns are missing from the dataframe: {missing}")

    # Define control columns
    control_cols = ['wind', 'min', 'category', 'year', 'elapsedyrs', 'log_ndam15', 'source_code']

    # Prepare design matrices
    y = df['log_alldeaths'].astype(float)

    # Model 1: continuous masfem
    X1 = df[['masfem_std'] + control_cols].astype(float)
    X1 = sm.add_constant(X1)
    model_masfem_cont = sm.OLS(y, X1).fit(cov_type='HC3')

    # Model 2: binary gender indicator
    X2 = df[['gender_mf'] + control_cols].astype(float)
    X2 = sm.add_constant(X2)
    model_gender_bin = sm.OLS(y, X2).fit(cov_type='HC3')

    # Model 3: both measures included
    X3 = df[['masfem_std', 'gender_mf'] + control_cols].astype(float)
    X3 = sm.add_constant(X3)
    model_both = sm.OLS(y, X3).fit(cov_type='HC3')

    # Return the fitted results for further inspection (e.g., .summary())
    return {
        'model_masfem_cont': model_masfem_cont,
        'model_gender_bin': model_gender_bin,
        'model_both': model_both
    }


