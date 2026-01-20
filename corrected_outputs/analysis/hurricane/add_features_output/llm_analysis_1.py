from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/hurricane/add_features_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into the final analysis dataframe.
    Produces the following new columns used by the model:
      - log_deaths: np.log1p(alldeaths)
      - masfem_center: masfem centered around its sample mean
      - year_center: year centered around its sample mean
    Drops rows with missing values in the variables required for the primary analysis.
    """
    df = df.copy()

    # Ensure the key columns exist and coerce to numeric where appropriate
    required = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'year']
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in input dataframe")

    # Coerce to numeric (will create NaNs for non-numeric values) and drop rows missing any required column
    df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce')
    df['masfem'] = pd.to_numeric(df['masfem'], errors='coerce')
    df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce')
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    df['min'] = pd.to_numeric(df['min'], errors='coerce')
    df['category'] = pd.to_numeric(df['category'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    df = df.dropna(subset=required)

    # Dependent variable: log(1 + deaths) to reduce skew and include zeros
    df['log_deaths'] = np.log1p(df['alldeaths'].astype(float))

    # Independent variables: center the continuous masfem and year variables
    df['masfem_center'] = df['masfem'] - df['masfem'].mean()
    df['year_center'] = df['year'] - df['year'].mean()

    # Ensure gender_mf is binary 0/1
    df['gender_mf'] = df['gender_mf'].astype(int)

    # Keep only the columns needed for modeling (but preserve original columns as well)
    # Final dataframe will still include original columns plus the derived ones
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit primary and robustness models testing whether more feminine hurricane names are associated with higher fatalities.

    Primary model:
      - OLS regression on log_deaths with robust (HC3) standard errors:
        log_deaths ~ masfem_center + gender_mf + wind + min + category + year_center

    Robustness:
      - Negative binomial regression on raw counts (alldeaths) with the same covariates to respect count nature of the DV.

    Returns a dict with keys 'ols' and 'neg_binom' containing fitted results objects. If NB fails, neg_binom will be None.
    """
    results = {}

    # Prepare design matrices
    covariates = ['masfem_center', 'gender_mf', 'wind', 'min', 'category', 'year_center']
    for c in covariates:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in dataframe passed to model")

    X = df[covariates].astype(float)
    X = sm.add_constant(X)
    y = df['log_deaths'].astype(float)

    # Primary: OLS on logged deaths with robust standard errors (HC3)
    ols_res = sm.OLS(y, X).fit(cov_type='HC3')
    results['ols'] = ols_res

    # Robustness: Negative binomial on raw counts
    # Use statsmodels' GLM with NegativeBinomial family (may require good starting values; wrap in try/except)
    try:
        import statsmodels.formula.api as smf
        # Build a formula string for convenience
        formula = 'alldeaths ~ masfem_center + gender_mf + wind + min + category + year_center'
        nb_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial()).fit()
        results['neg_binom'] = nb_model
    except Exception as e:
        # If NB fails (convergence or other issues), return None for that entry and keep OLS result
        results['neg_binom'] = None
        results['neg_binom_error'] = str(e)

    return results


