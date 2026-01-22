from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/replace_and_positive_statement_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataset into a dataframe ready for modeling.

    Produces the following new/clean columns used by the model:
      - alldeaths_count: integer count of total deaths (from 'alldeaths')
      - log_alldeaths: log(1 + alldeaths)
      - masfem_z: standardized (z-scored) 'masfem' variable
      - gender_mf: ensured integer 0/1 indicator for female name (original column kept but coerced)
      - source: coerced to string/categorical for use with C(source) in formulas

    Rows missing any of the required columns (masfem, gender_mf, alldeaths, wind, category, min, year, elapsedyrs, source)
    are dropped since they are required for the primary models below.
    """

    df = df.copy()

    # Ensure required columns exist
    required = ['masfem', 'gender_mf', 'alldeaths', 'wind', 'category', 'min', 'year', 'elapsedyrs', 'source']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows with missingness in key variables used for primary specification
    df = df.dropna(subset=required)

    # Force numeric types where appropriate
    df['alldeaths_count'] = pd.to_numeric(df['alldeaths'], errors='coerce').fillna(0).astype(int)
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    df['category'] = pd.to_numeric(df['category'], errors='coerce')
    df['min'] = pd.to_numeric(df['min'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')

    # Re-drop rows that turned NA after coercion
    df = df.dropna(subset=['alldeaths_count', 'wind', 'category', 'min', 'year', 'elapsedyrs'])

    # Log transform of fatalities (use log1p to handle zeros)
    df['log_alldeaths'] = np.log1p(df['alldeaths_count'])

    # Standardize masfem (z-score). Keep original masfem as well if needed.
    df['masfem'] = pd.to_numeric(df['masfem'], errors='coerce')
    df = df.dropna(subset=['masfem'])
    mas_mean = df['masfem'].mean()
    mas_std = df['masfem'].std(ddof=0)
    if mas_std == 0 or np.isnan(mas_std):
        # If no variation, keep zeros (but downstream models will not be informative)
        df['masfem_z'] = 0.0
    else:
        df['masfem_z'] = (df['masfem'] - mas_mean) / mas_std

    # Ensure binary gender indicator is integer 0/1
    df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce').astype(int)

    # Coerce source to categorical/string for use with C(source) in formulas
    df['source'] = df['source'].astype(str)

    # Final sanity check: keep rows with non-negative death counts
    df = df[df['alldeaths_count'] >= 0]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit two complementary models testing whether more feminine hurricane names are associated
    with higher fatalities (proxying fewer precautions / lower perceived threat):

    1) Negative Binomial GLM on the count of fatalities (alldeaths_count) to respect count nature
       and potential overdispersion.
    2) OLS on log(1 + fatalities) as a robust-sense continuous comparison.

    Both models control for storm intensity/severity (wind, category, min), year, elapsedyrs,
    and source fixed effects (C(source)). Robust (HC3) standard errors are used.

    Returns a dictionary with fitted model objects and textual summaries.
    """

    import statsmodels.formula.api as smf

    # Ensure the input df has the transformed columns
    required_cols = ['alldeaths_count', 'log_alldeaths', 'masfem_z', 'gender_mf', 'wind', 'category', 'min', 'year', 'elapsedyrs', 'source']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing columns required for modeling: {missing}")

    # Define formulas (include source as categorical fixed effects)
    formula_nb = 'alldeaths_count ~ masfem_z + gender_mf + wind + category + min + year + elapsedyrs + C(source)'
    formula_ols = 'log_alldeaths ~ masfem_z + gender_mf + wind + category + min + year + elapsedyrs + C(source)'

    # Fit Negative Binomial via GLM (handles overdispersion vs Poisson)
    try:
        nb_model = smf.glm(formula=formula_nb, data=df, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
    except Exception as e:
        # Fallback: try Poisson with robust SE if NB fails
        nb_model = smf.glm(formula=formula_nb, data=df, family=sm.families.Poisson()).fit(cov_type='HC3')

    # Fit OLS on logged fatalities
    ols_model = smf.ols(formula=formula_ols, data=df).fit(cov_type='HC3')

    results = {
        'nb_model': nb_model,
        'ols_model': ols_model,
        'nb_summary_text': nb_model.summary().as_text(),
        'ols_summary_text': ols_model.summary().as_text(),
        # helpful small diagnostics
        'n_obs': int(df.shape[0]),
        'mean_alldeaths': float(df['alldeaths_count'].mean()),
        'median_alldeaths': float(df['alldeaths_count'].median())
    }

    return results


