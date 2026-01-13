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
    Transform the original hurricane dataframe to produce the columns used in the statistical model.

    Produces:
    - alldeaths (kept from input) -- dependent variable (counts)
    - masfem_z (standardized masfem) -- main independent variable
    - gender_female (0/1, from gender_mf) -- binary gender indicator control
    - wind, min, category (category cast to categorical)
    - year_center (year centered at mean)
    - log_ndam15 (log(ndam15 + 1))
    - source (kept as categorical)

    Rows with missing values in core variables are dropped.
    """
    df = df.copy()

    # Ensure numeric columns are numeric
    numeric_cols = ['alldeaths', 'masfem', 'wind', 'min', 'category', 'year', 'ndam15', 'gender_mf']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Keep only rows with the core variables needed for the primary model
    required = ['alldeaths', 'masfem', 'wind', 'min', 'category', 'year']
    existing_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=existing_required)

    # Dependent variable: ensure non-negative integer counts
    df['alldeaths'] = df['alldeaths'].astype(float)
    # (We keep floats for modeling but they represent counts.)

    # Independent variable: standardize masfem (z-score)
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # Binary female indicator (control)
    if 'gender_mf' in df.columns:
        # gender_mf already encoded as 0/1 in the raw data (0 male, 1 female)
        df['gender_female'] = df['gender_mf'].astype(int)
    else:
        # if absent, create NA column
        df['gender_female'] = np.nan

    # Log-transformed damage (proxy for exposure / economic scale)
    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log(df['ndam15'].fillna(0) + 1)
    else:
        df['log_ndam15'] = np.nan

    # Year centered
    df['year_center'] = df['year'] - df['year'].mean()

    # Category and source as categorical
    df['category'] = df['category'].astype('category')
    if 'source' in df.columns:
        df['source'] = df['source'].astype('category')
    else:
        df['source'] = 'unknown'

    # Final column list check: keep columns used in modeling plus original alldeaths
    model_cols = ['alldeaths', 'masfem_z', 'gender_female', 'wind', 'min', 'category', 'year_center', 'log_ndam15', 'source']
    existing_model_cols = [c for c in model_cols if c in df.columns]

    # Drop rows with missing values in any of the model columns (conservative approach)
    df = df.dropna(subset=existing_model_cols)

    # Return dataframe including at least the columns listed above (plus any others present)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a negative binomial generalized linear model predicting hurricane fatalities (alldeaths)
    from the standardized femininity rating of the hurricane name (masfem_z), controlling for
    storm intensity and temporal/source covariates.

    Model formula:
      alldeaths ~ masfem_z + gender_female + wind + min + C(category) + year_center + log_ndam15 + C(source)

    Returns a statsmodels results object with robust (HC3) standard errors applied.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Define formula using the transformed column names
    formula = (
        'alldeaths ~ masfem_z + gender_female + wind + min + C(category) '
        '+ year_center + log_ndam15 + C(source)'
    )

    # Fit Negative Binomial GLM (appropriate for over-dispersed count data)
    model_glm = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial())
    res = model_glm.fit()

    # Obtain robust covariance (HC3) for inference
    try:
        res_robust = res.get_robustcov_results(cov_type='HC3')
    except Exception:
        # Fall back to the original results if robust cov cannot be computed
        res_robust = res

    # Print a short summary for quick inspection (caller can still examine the returned object)
    print(res_robust.summary())

    return res_robust


