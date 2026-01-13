from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/replace_with_rvs_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Columns required for the primary analyses
    required_cols = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'year']

    # Ensure numeric where expected; coerce non-numeric to NaN
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing any required column
    df = df.dropna(subset=required_cols)

    # Create centered masculinity-femininity variable (masfem_c)
    df['masfem_c'] = df['masfem'] - df['masfem'].mean()

    # Keep raw count outcome and create log(1 + x) transformed version for OLS robustness
    df['alldeaths'] = df['alldeaths'].astype(int)
    df['log_alldeaths'] = np.log1p(df['alldeaths'])

    # Standardize continuous controls to improve numeric stability
    # Use population std (ddof=0) for consistency
    df['wind_z'] = (df['wind'] - df['wind'].mean()) / df['wind'].std(ddof=0)
    df['min_z'] = (df['min'] - df['min'].mean()) / df['min'].std(ddof=0)

    # Center year to aid interpretation
    df['year_c'] = df['year'] - df['year'].mean()

    # Convert category to categorical and create dummy indicators (drop_first=True to use first category as reference)
    df['category_cat'] = df['category'].astype('category')
    cat_dummies = pd.get_dummies(df['category_cat'], prefix='cat', drop_first=True)
    # Concatenate dummies (these will create columns like 'cat_2','cat_3', etc. depending on data)
    df = pd.concat([df, cat_dummies], axis=1)

    # Ensure gender_mf is numeric 0/1
    df['gender_mf'] = df['gender_mf'].astype(int)

    # Final dataframe returned contains all variables used by the model
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    # This modeling function assumes df is the output of transform(df)
    # Build the list of exogenous (predictor) variables to include in the models
    # Include masfem_c (IV), controls: gender_mf, wind_z, min_z, year_c and any category dummies created
    category_cols = [c for c in df.columns if c.startswith('cat_')]
    exog_vars = ['masfem_c', 'gender_mf', 'wind_z', 'min_z', 'year_c'] + category_cols

    # Ensure exog exists
    X = df[exog_vars]
    X = sm.add_constant(X)

    # Dependent variables
    y_count = df['alldeaths']          # count outcome for NB / Poisson
    y_log = df['log_alldeaths']       # transformed outcome for OLS robustness

    # 1) Negative binomial GLM for count outcome (primary model because alldeaths is overdispersed count)
    nb_model = sm.GLM(y_count, X, family=sm.families.NegativeBinomial()).fit()

    # 2) OLS on log(1 + deaths) as a robustness check
    ols_model = sm.OLS(y_log, X).fit()

    # Return the fitted result objects so the caller can inspect summaries, coefficients, CIs, etc.
    return {
        'nb_model': nb_model,
        'ols_model': ols_model,
        'exog_vars': exog_vars
    }


