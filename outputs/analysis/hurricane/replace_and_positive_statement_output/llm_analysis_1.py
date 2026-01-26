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
    # Work on a copy
    df = df.copy()

    # Ensure columns exist
    required_cols = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'year', 'elapsedyrs', 'source', 'masfem_mturk']
    # Some columns may be missing in particular data exports; guard against KeyError by creating if absent
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Convert source to category for use in formula-based models
    df['source'] = df['source'].astype('category')

    # Drop rows missing the core dependent or primary IV or key severity controls
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind', 'min', 'category'])

    # Create logged fatalities for OLS/robustness (add 1 to keep zeros)
    df['log_alldeaths'] = np.log(df['alldeaths'] + 1)

    # Standardize the masfem measures (z-scores) to make coefficients comparable
    df['z_masfem'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1)

    # Also standardize the MTurk-based rating (used in robustness checks)
    if df['masfem_mturk'].notna().sum() > 0:
        df['z_masfem_mturk'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1)
    else:
        df['z_masfem_mturk'] = np.nan

    # Ensure category, year, elapsedyrs are numeric
    df['category'] = pd.to_numeric(df['category'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')

    # Drop rows with missing values in controls after coercion
    df = df.dropna(subset=['wind', 'min', 'category', 'year', 'elapsedyrs'])

    # Keep the columns needed for modeling explicit and return
    model_cols = ['alldeaths', 'log_alldeaths', 'masfem', 'z_masfem', 'masfem_mturk', 'z_masfem_mturk', 'gender_mf', 'wind', 'min', 'category', 'year', 'elapsedyrs', 'source']
    # Some columns may not have been filled (masfem_mturk), but keep them in df so modeling code can handle NA appropriately
    return df[model_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    import statsmodels.api as _sm
    import statsmodels.formula.api as smf

    results = {}

    # Formula for primary specification: test whether name femininity predicts fatalities controlling for storm severity and time/source
    # Negative binomial on counts (alldeaths) is appropriate for overdispersed count data.
    nb_formula = 'alldeaths ~ z_masfem + gender_mf + wind + min + category + year + elapsedyrs + C(source)'

    # Fit Negative Binomial (GLM). Use robust (HC3) covariances for inference.
    try:
        nb_model = smf.glm(nb_formula, data=df, family=_sm.families.NegativeBinomial()).fit(cov_type='HC3')
        results['nb_primary'] = nb_model
    except Exception as e:
        results['nb_primary_error'] = str(e)

    # OLS robustness: logged fatalities
    ols_formula = 'log_alldeaths ~ z_masfem + gender_mf + wind + min + category + year + elapsedyrs + C(source)'
    try:
        ols_model = smf.ols(ols_formula, data=df).fit(cov_type='HC3')
        results['ols_log_fatalities'] = ols_model
    except Exception as e:
        results['ols_error'] = str(e)

    # Robustness: use MTurk-based femininity rating instead of masfem if available
    if df['z_masfem_mturk'].notna().sum() > 0:
        nb_formula_mturk = 'alldeaths ~ z_masfem_mturk + gender_mf + wind + min + category + year + elapsedyrs + C(source)'
        try:
            nb_model_mturk = smf.glm(nb_formula_mturk, data=df.dropna(subset=['z_masfem_mturk']), family=_sm.families.NegativeBinomial()).fit(cov_type='HC3')
            results['nb_mturk'] = nb_model_mturk
        except Exception as e:
            results['nb_mturk_error'] = str(e)

    # Additional diagnostic: simple tabulation of mean fatalities by gender_mf
    try:
        results['means_by_gender'] = df.groupby('gender_mf')['alldeaths'].agg(['count', 'mean', 'median']).to_dict()
    except Exception:
        results['means_by_gender'] = None

    # Return fitted model result objects (statsmodels results). The caller can call .summary() on each result.
    return results


