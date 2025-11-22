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
    Transform the raw hurricane dataframe into a modelling-ready dataframe.

    Adds/returns the following columns (exact names used by the model):
      - log_alldeaths : np.log(alldeaths + 1)
      - masfem_z      : z-scored version of masfem (continuous femininity rating)
      - IsFemaleName  : integer 0/1 from gender_mf
      - masfem_mturk_z: z-scored masfem_mturk (alternative femininity measure)
      - log_ndam15    : np.log(ndam15 + 1) (alternative outcome: log damage)
      - year_center   : year centered about the sample mean
    The function also drops rows with missing values in the variables needed for the primary models.
    """

    df = df.copy()

    # Ensure expected raw columns exist and convert numeric where appropriate
    # Convert columns to numeric if they are not already
    numeric_cols = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'category', 'min', 'elapsedyrs', 'year', 'ndam15', 'masfem_mturk']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values in primary IV, DV, and key controls
    required_for_main = ['alldeaths', 'masfem', 'wind', 'category', 'min', 'elapsedyrs', 'year']
    df = df.dropna(subset=required_for_main)

    # Dependent variable: log-transformed fatalities (add 1 to handle zeros)
    df['log_alldeaths'] = np.log(df['alldeaths'] + 1)

    # Alternative dependent: logged damage (2015-normalized). Keep if present.
    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log(df['ndam15'].fillna(0) + 1)
    else:
        df['log_ndam15'] = np.nan

    # Independent variables
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)
    # Alternative continuous femininity from MTurk ratings if present
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1.0)
    else:
        df['masfem_mturk_z'] = np.nan

    # Binary female-name indicator (column gender_mf is already 0/1 in the schema; coerce to int)
    df['IsFemaleName'] = df['gender_mf'].fillna(0).astype(int)

    # Center year to help interpretation
    df['year_center'] = df['year'] - df['year'].mean()

    # Keep only columns needed for modeling + original identifiers
    model_cols = [
        'ind', 'year', 'year_center', 'alldeaths', 'log_alldeaths', 'ndam15', 'log_ndam15',
        'masfem', 'masfem_z', 'masfem_mturk_z', 'gender_mf', 'IsFemaleName',
        'wind', 'category', 'min', 'elapsedyrs', 'name', 'source'
    ]
    # Some of these may not be present in the raw df; select intersection
    keep_cols = [c for c in model_cols if c in df.columns]
    df = df[keep_cols]

    # Final sanity: drop rows with infinite or NaN in 'log_alldeaths' or masfem_z or main controls
    final_required = ['log_alldeaths', 'masfem_z', 'wind', 'category', 'min', 'elapsedyrs', 'year_center']
    df = df.dropna(subset=final_required)

    # Reset index
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit models to estimate the relationship between hurricane name femininity and outcomes.

    Returns a dictionary of fitted model results objects (statsmodels results instances):
      - ols_log_deaths : OLS on log(alldeaths+1) with robust (HC3) SEs
      - nb_deaths      : Negative Binomial GLM on raw alldeaths (counts)
      - ols_log_damage : OLS on log(ndam15+1) as an alternative outcome (if ndam15 present)

    Model specification (same covariates across models):
      Outcome ~ masfem_z + IsFemaleName + wind + category + min + elapsedyrs + year_center
    """

    import statsmodels.api as sm

    results = {}

    # Ensure the dataframe used here is the transformed df (has the expected columns)
    required_cols = ['log_alldeaths', 'masfem_z', 'IsFemaleName', 'wind', 'category', 'min', 'elapsedyrs', 'year_center']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build exogenous matrix
    exog_cols = ['masfem_z', 'IsFemaleName', 'wind', 'category', 'min', 'elapsedyrs', 'year_center']
    X = df[exog_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # 1) OLS on logged fatalities
    y_ols = df['log_alldeaths'].astype(float)
    ols_model = sm.OLS(y_ols, X)
    ols_res = ols_model.fit(cov_type='HC3')  # robust SEs (HC3)
    results['ols_log_deaths'] = ols_res

    # 2) Negative Binomial on raw counts of fatalities (robust to overdispersion)
    # Use the same exogenous variables; endog should be integer counts
    if 'alldeaths' in df.columns:
        y_counts = df['alldeaths'].astype(float)
        try:
            nb_model = sm.GLM(y_counts, X, family=sm.families.NegativeBinomial())
            nb_res = nb_model.fit()
            results['nb_deaths'] = nb_res
        except Exception as e:
            # If NB fails, return the exception message instead of a result object
            results['nb_deaths'] = {'error': str(e)}
    else:
        results['nb_deaths'] = {'error': 'alldeaths column not present'}

    # 3) Robustness: OLS on logged damage (ndam15) if available
    if 'log_ndam15' in df.columns and df['log_ndam15'].notna().any():
        y_dmg = df['log_ndam15'].astype(float)
        ols_dmg = sm.OLS(y_dmg, X).fit(cov_type='HC3')
        results['ols_log_damage'] = ols_dmg
    else:
        results['ols_log_damage'] = None

    return results


