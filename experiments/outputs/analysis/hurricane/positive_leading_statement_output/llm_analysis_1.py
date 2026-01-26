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
    # Make a copy
    df = df.copy()

    # Ensure key numeric columns are numeric
    numeric_cols = ['alldeaths', 'masfem', 'masfem_mturk', 'wind', 'min', 'category', 'ndam15', 'year', 'gender_mf']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the primary DV or primary IV
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind', 'category', 'year'])

    # Create log-transform of deaths for OLS robustness
    df['log_alldeaths'] = np.log(df['alldeaths'] + 1)

    # Standardize masfem measures (z-scores) for interpretability
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1)
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1)
    else:
        df['masfem_mturk_z'] = np.nan

    # Log transform of normalized damages (ndam15) to capture scale/exposure and reduce skew
    if 'ndam15' in df.columns:
        df['ndam15_log'] = np.log(df['ndam15'].fillna(0) + 1)
    else:
        df['ndam15_log'] = np.nan

    # Center year to improve interpretability and reduce collinearity with intercept
    df['year_c'] = df['year'] - df['year'].mean()

    # Ensure binary gender_mf is 0/1
    if 'gender_mf' in df.columns:
        df['gender_mf'] = df['gender_mf'].apply(lambda x: 1 if x == 1 else 0)
    else:
        df['gender_mf'] = np.nan

    # Keep only columns needed for modeling (but return full df with derived columns)
    # Return df with the derived columns used in modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Build modeling dataframe: drop rows missing any predictors needed for main model
    cols_needed = ['alldeaths', 'masfem_z', 'wind', 'min', 'category', 'ndam15_log', 'year_c', 'gender_mf', 'log_alldeaths', 'masfem_mturk_z']
    model_df = df.dropna(subset=['alldeaths', 'masfem_z', 'wind', 'min', 'category', 'ndam15_log', 'year_c'])

    results = {}

    # 1) Main model: Negative Binomial regression on raw death counts
    # Formula includes main IV (masfem_z), the binary gender label, and controls for intensity, damage and year
    formula_nb = 'alldeaths ~ masfem_z + gender_mf + wind + min + category + ndam15_log + year_c'
    try:
        nb_model = smf.glm(formula=formula_nb, data=model_df, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
        results['neg_binomial'] = nb_model
    except Exception as e:
        results['neg_binomial_error'] = str(e)

    # 2) Robustness: OLS on log(alldeaths + 1)
    formula_ols = 'log_alldeaths ~ masfem_z + gender_mf + wind + min + category + ndam15_log + year_c'
    try:
        ols_model = smf.ols(formula=formula_ols, data=model_df).fit(cov_type='HC3')
        results['ols_log_deaths'] = ols_model
    except Exception as e:
        results['ols_error'] = str(e)

    # 3) Robustness: Poisson regression with robust SEs (check for overdispersion relative to Poisson)
    formula_pois = formula_nb
    try:
        pois_model = smf.glm(formula=formula_pois, data=model_df, family=sm.families.Poisson()).fit(cov_type='HC3')
        results['poisson'] = pois_model
    except Exception as e:
        results['poisson_error'] = str(e)

    # 4) Robustness: use masfem_mturk_z (if available) as alternative IV
    if model_df['masfem_mturk_z'].notna().sum() > 10:
        try:
            formula_alt = 'alldeaths ~ masfem_mturk_z + gender_mf + wind + min + category + ndam15_log + year_c'
            nb_model_alt = smf.glm(formula=formula_alt, data=model_df.dropna(subset=['masfem_mturk_z']), family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
            results['neg_binomial_mturk'] = nb_model_alt

            formula_ols_alt = 'log_alldeaths ~ masfem_mturk_z + gender_mf + wind + min + category + ndam15_log + year_c'
            ols_model_alt = smf.ols(formula=formula_ols_alt, data=model_df.dropna(subset=['masfem_mturk_z'])).fit(cov_type='HC3')
            results['ols_log_deaths_mturk'] = ols_model_alt
        except Exception as e:
            results['mturk_error'] = str(e)
    else:
        results['mturk_message'] = 'Insufficient masfem_mturk data for robustness check.'

    # 5) Interaction test: does the effect of name femininity differ when the name was labeled female vs male?
    formula_int = 'alldeaths ~ masfem_z * gender_mf + wind + min + category + ndam15_log + year_c'
    try:
        nb_model_int = smf.glm(formula=formula_int, data=model_df, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
        results['neg_binomial_interaction'] = nb_model_int
    except Exception as e:
        results['interaction_error'] = str(e)

    # Return results dictionary (each value is a fitted statsmodels object or error/message string)
    return results


