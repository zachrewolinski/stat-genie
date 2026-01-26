from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/add_features_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the hurricane dataset for analysis.

    Transformations performed:
    - Drop rows missing the core variables needed for analysis.
    - Create standardized femininity score (masfem_std) from 'masfem'.
    - Ensure binary gender indicator 'gender_mf' is integer 0/1.
    - Create log-transformed 2015-normalized damage: log_ndam15 = log(ndam15 + 1).
    - Create dummies for 'source' (prefix 'source_').
    - Keep only columns necessary for modeling but return full dataframe with new columns added.
    """
    # Copy to avoid modifying original in place
    df = df.copy()

    # Ensure key numeric columns exist; if not, this will raise an informative KeyError
    required_cols = ['masfem', 'gender_mf', 'alldeaths', 'ndam15', 'wind', 'category', 'min', 'year', 'elapsedyrs', 'source']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing required columns in input dataframe: {missing}")

    # Drop rows with missing values in core variables for primary analyses
    df = df.dropna(subset=['masfem', 'alldeaths', 'ndam15', 'wind', 'category', 'min', 'year', 'elapsedyrs', 'gender_mf'])

    # Standardize masfem for easier interpretation of coefficients
    df['masfem'] = pd.to_numeric(df['masfem'], errors='coerce')
    masfem_mean = df['masfem'].mean()
    masfem_stddev = df['masfem'].std(ddof=0)
    # Guard against zero std
    if masfem_stddev == 0 or np.isnan(masfem_stddev):
        df['masfem_std'] = df['masfem'] - masfem_mean
    else:
        df['masfem_std'] = (df['masfem'] - masfem_mean) / masfem_stddev

    # Ensure binary indicator is integer 0/1
    # Some datasets use 0/1 floats; coerce to integer
    df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce').astype(int)

    # Log transform damage (ndam15) as it's heavily right skewed
    df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce').fillna(0.0)
    df['log_ndam15'] = np.log(df['ndam15'] + 1.0)

    # Ensure alldeaths is numeric integer (count)
    df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce').fillna(0).astype(int)

    # Coerce other numeric control variables to numeric types
    for c in ['wind', 'category', 'min', 'year', 'elapsedyrs']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Create dummies for 'source' to control for source-specific systematic differences
    # Keep all dummies (drop_first=False) so modeler can drop a baseline manually if desired
    source_dummies = pd.get_dummies(df['source'].astype(str), prefix='source')
    # Attach dummies to dataframe
    if not source_dummies.empty:
        # Align index and concat
        df = pd.concat([df, source_dummies], axis=1)

    # Optionally, drop any rows that still contain NaNs in the modeling columns
    modeling_cols = ['masfem_std', 'gender_mf', 'alldeaths', 'log_ndam15', 'wind', 'category', 'min', 'year', 'elapsedyrs']
    # include any source_ columns present
    modeling_cols += [c for c in df.columns if c.startswith('source_')]
    df = df.dropna(subset=modeling_cols)

    # Return the transformed dataframe. It contains at least the columns specified in the conceptual variables.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit models testing whether more feminine hurricane names are associated with fewer precautions
    (proxied by more fatalities or more damage).

    Models run:
    1) Negative binomial GLM predicting alldeaths (counts) from feminine name score and controls.
    2) OLS predicting log(ndam15 + 1) from feminine name score and controls.

    Returns a dictionary containing fitted model objects and textual summaries.
    """
    import statsmodels.api as sm

    # Copy and make sure required transformed columns exist
    df = df.copy()
    required = ['masfem_std', 'gender_mf', 'alldeaths', 'log_ndam15', 'wind', 'category', 'min', 'year', 'elapsedyrs']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Dataframe missing required transformed columns: {missing}")

    # Build list of covariates
    base_controls = ['wind', 'category', 'min', 'year', 'elapsedyrs']
    # include any source dummies created in transform
    source_cols = [c for c in df.columns if c.startswith('source_')]

    predictors = ['masfem_std', 'gender_mf'] + base_controls + source_cols

    # Prepare design matrix X (add constant)
    X = df[predictors].astype(float)
    X = sm.add_constant(X)

    # 1) Negative binomial for fatalities (counts)
    y_deaths = df['alldeaths']
    # Use GLM NegativeBinomial (log link by default)
    try:
        nb_model = sm.GLM(y_deaths, X, family=sm.families.NegativeBinomial()).fit()
        nb_summary = nb_model.summary().as_text()
    except Exception as e:
        nb_model = None
        nb_summary = f"Negative binomial model failed: {str(e)}"

    # 2) OLS for log damage
    y_logdam = df['log_ndam15']
    try:
        ols_model = sm.OLS(y_logdam, X).fit()
        ols_summary = ols_model.summary().as_text()
    except Exception as e:
        ols_model = None
        ols_summary = f"OLS model failed: {str(e)}"

    # Return both models and summaries for inspection
    results = {
        'predictors': predictors,
        'nb_model': nb_model,
        'nb_summary': nb_summary,
        'ols_model': ols_model,
        'ols_summary': ols_summary
    }
    return results


