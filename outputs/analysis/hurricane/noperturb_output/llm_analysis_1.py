from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/noperturb_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into a modeling-ready dataframe.

    Creates/ensures these final columns used in the models:
      - masfem_z: standardized masfem (z-score)
      - gender_mf: binary indicator (0/1) for female name
      - alldeaths: observed fatalities (numeric)
      - ndam15: inflation/wealth/population-normalized damage (numeric)
      - log_ndam15: log(1 + ndam15)
      - wind, category, min, elapsedyrs, year_centered, source: controls

    Drops rows with missing values in core variables.
    """
    df = df.copy()

    # Ensure numeric columns are numeric where present
    numeric_cols = ['masfem', 'gender_mf', 'wind', 'category', 'min', 'alldeaths', 'ndam15', 'elapsedyrs', 'year']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Convert source to categorical if present
    if 'source' in df.columns:
        df['source'] = df['source'].astype('category')

    # Drop rows missing the key predictors/outcomes/controls
    required = [c for c in ['masfem', 'alldeaths', 'ndam15', 'wind', 'category', 'min', 'year'] if c in df.columns]
    if len(required) > 0:
        df = df.dropna(subset=required).reset_index(drop=True)

    # Standardize masfem (z-score). Use population std (ddof=0) for stability.
    if 'masfem' in df.columns:
        mu = df['masfem'].mean()
        sigma = df['masfem'].std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            df['masfem_z'] = 0.0
        else:
            df['masfem_z'] = (df['masfem'] - mu) / sigma

    # Ensure gender_mf is integer 0/1 when present
    if 'gender_mf' in df.columns:
        # coerce any non-0/1 values to NaN first, then to int where possible
        df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce')
        # If there are NaNs in gender_mf after coercion, leave them as-is (they will be dropped already if required)
        df.loc[df['gender_mf'].notnull(), 'gender_mf'] = df.loc[df['gender_mf'].notnull(), 'gender_mf'].astype(int)

    # Outcomes: log transform damages for OLS model
    if 'ndam15' in df.columns:
        # Ensure non-negative
        df['ndam15'] = df['ndam15'].clip(lower=0)
        df['log_ndam15'] = np.log1p(df['ndam15'])

    # Also create a logged deaths variable for exploratory use
    if 'alldeaths' in df.columns:
        df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce').fillna(0).astype(int)
        df['log_alldeaths'] = np.log1p(df['alldeaths'])

    # Center year to control for linear time trend
    if 'year' in df.columns:
        df['year_centered'] = df['year'] - df['year'].mean()

    # Final safety: drop any remaining rows with NaNs in model columns
    needed_cols = ['masfem_z', 'alldeaths', 'log_ndam15', 'wind', 'category', 'min', 'elapsedyrs', 'year_centered', 'source', 'gender_mf']
    # keep only columns that actually exist in df for the final dropna check
    needed_cols = [c for c in needed_cols if c in df.columns]
    if len(needed_cols) > 0:
        df = df.dropna(subset=needed_cols).reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two models to test whether more-feminine hurricane names are associated with fewer precautionary outcomes
    (operationalized as higher fatalities and higher damages):

    1) Negative Binomial regression for alldeaths (count outcome):
       alldeaths ~ masfem_z + gender_mf + wind + category + min + elapsedyrs + year_centered + C(source)

    2) OLS regression for log_ndam15 (continuous):
       log_ndam15 ~ masfem_z + gender_mf + wind + category + min + elapsedyrs + year_centered + C(source)

    Returns a dict with both fitted model results.
    """
    import statsmodels.formula.api as smf

    results = {}
    df = df.copy()

    # Ensure categorical encoding for source in formula; if source absent, remove from formula
    source_term = ' + C(source)' if 'source' in df.columns else ''

    # Specify formulas
    formula_nb = f"alldeaths ~ masfem_z + gender_mf + wind + category + min + elapsedyrs + year_centered{source_term}"
    formula_ols = f"log_ndam15 ~ masfem_z + gender_mf + wind + category + min + elapsedyrs + year_centered{source_term}"

    # Fit Negative Binomial for deaths (handles count data with overdispersion)
    try:
        nb_model = smf.glm(formula_nb, data=df, family=sm.families.NegativeBinomial()).fit()
        results['deaths_model'] = nb_model
        print('\nNegative Binomial model for alldeaths fitted successfully. Summary:')
        print(nb_model.summary())
    except Exception as e:
        results['deaths_model'] = None
        print('\nFailed to fit Negative Binomial model for alldeaths: ', e)

    # Fit OLS for logged damages with robust standard errors
    try:
        ols_model = smf.ols(formula_ols, data=df).fit(cov_type='HC3')
        results['damage_model'] = ols_model
        print('\nOLS model for log_ndam15 fitted successfully. Summary:')
        print(ols_model.summary())
    except Exception as e:
        results['damage_model'] = None
        print('\nFailed to fit OLS model for log_ndam15: ', e)

    return results


