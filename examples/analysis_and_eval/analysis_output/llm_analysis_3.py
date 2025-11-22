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
    Transform the raw hurricane dataset into analysis-ready dataframe.

    Produces these columns (used in modeling):
      - masfem_std: standardized (z-scored) masfem score
      - gender_female: binary 0/1 mapping of gender_mf
      - alldeaths: raw death counts (integer)
      - log_alldeaths: log1p(alldeaths) for OLS robustness
      - wind, min, category: kept as provided (category kept as categorical later)
      - year_center: year centered around its mean
      - source: categorical source column

    Rows with missing values in variables required for the main analyses are dropped.
    """
    # work on a copy
    df = df.copy()

    # Ensure expected columns exist
    required_cols = ['masfem', 'gender_mf', 'alldeaths', 'wind', 'min', 'category', 'year', 'source']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Input dataframe is missing required columns: {missing}")

    # Drop rows missing primary IV or DV or core controls.
    df = df.dropna(subset=['masfem', 'alldeaths', 'wind', 'min', 'category', 'year', 'source'])

    # Ensure numeric types where expected
    df['masfem'] = pd.to_numeric(df['masfem'], errors='coerce')
    df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce').astype('Int64')
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    df['min'] = pd.to_numeric(df['min'], errors='coerce')
    df['category'] = pd.to_numeric(df['category'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    # Re-drop any rows made NA by coercion
    df = df.dropna(subset=['masfem', 'alldeaths', 'wind', 'min', 'category', 'year', 'source'])

    # Standardize masfem for interpretability
    mas_mean = df['masfem'].mean()
    mas_std = df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0
    df['masfem_std'] = (df['masfem'] - mas_mean) / mas_std

    # Binary female name indicator (explicitly name it gender_female for modeling)
    # gender_mf is coded 0 = male, 1 = female per schema; cast to int to be safe
    df['gender_female'] = df['gender_mf'].astype(int)

    # Dependent variable: keep raw count and also provide log(1 + count) for OLS robustness
    # convert to plain int (numpy) for modeling
    df['alldeaths'] = df['alldeaths'].astype(int)
    df['log_alldeaths'] = np.log1p(df['alldeaths'])

    # Year centering (helps interpretation of intercept and reduces collinearity)
    df['year_center'] = df['year'] - df['year'].mean()

    # Ensure source is categorical
    df['source'] = df['source'].astype('category')

    # Final safety: drop any infinite or NA rows that might remain after transforms
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['masfem_std', 'alldeaths', 'log_alldeaths', 'wind', 'min', 'category', 'year_center', 'source'])

    # Return the dataframe that contains the exact column names used by the model
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run primary Negative Binomial regression (counts) and an OLS robustness regression on log fatalities.

    Models:
      1) Negative Binomial GLM: alldeaths ~ masfem_std + wind + min + C(category) + year_center + C(source)
         (interprets coefficients as multiplicative effects on expected count)
      2) OLS on log1p(alldeaths) with same regressors for robustness.

    Returns a dictionary with 'nb_model' and 'ols_model' fitted result objects.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Verify required model columns are present
    model_cols = ['alldeaths', 'log_alldeaths', 'masfem_std', 'wind', 'min', 'category', 'year_center', 'source']
    missing = [c for c in model_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Dataframe missing columns required for modeling: {missing}")

    # Define formula (use categorical treatment for 'category' and 'source')
    formula = 'alldeaths ~ masfem_std + wind + min + C(category) + year_center + C(source)'
    formula_log = 'log_alldeaths ~ masfem_std + wind + min + C(category) + year_center + C(source)'

    # Fit Negative Binomial via GLM with NegativeBinomial family (handles overdispersion)
    try:
        nb_model = smf.glm(formula, data=df, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
    except Exception as e:
        # If GLM NegativeBinomial fails, fall back to statsmodels discrete NegativeBinomial
        try:
            nb_discrete = smf.negativebinomial(formula, data=df)  # may not exist in older statsmodels
            nb_model = nb_discrete.fit(disp=False)
        except Exception:
            # Last-resort: raise error with explanation
            raise RuntimeError(f"Failed to fit Negative Binomial model via GLM or discrete NB: {e}")

    # Fit OLS on log(1+alldeaths) as a robustness check
    ols_model = smf.ols(formula_log, data=df).fit(cov_type='HC3')

    # Return both fitted results
    return {
        'nb_model': nb_model,
        'ols_model': ols_model
    }


