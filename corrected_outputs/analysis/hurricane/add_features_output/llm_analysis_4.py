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
    Transform the raw hurricane dataframe into the analysis-ready dataframe.

    Adds/returns the following columns used in the model:
      - masfem_z: standardized (z-scored) masfem index
      - gender_female: binary indicator from gender_mf (1=female name, 0=male name)
      - log_deaths: log(alldeaths + 1)
      - log_ndam15: log(ndam15 + 1)
      - category: cast to categorical for modeling
    Also ensures wind, min, elapsedyrs, year are numeric and drops rows missing required variables.
    """
    df = df.copy()

    # Required raw columns (if any are missing we cannot run the analysis)
    required = ['masfem', 'gender_mf', 'alldeaths', 'ndam15', 'category', 'wind', 'min', 'elapsedyrs', 'year']
    # Drop rows with missing values in any required column
    df = df.dropna(subset=required)

    # Ensure numeric types
    for col in ['masfem', 'gender_mf', 'alldeaths', 'ndam15', 'wind', 'min', 'elapsedyrs', 'year', 'category']:
        # Attempt coercion where appropriate
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass

    # Re-drop if coercion produced new NA in required cols
    df = df.dropna(subset=required)

    # Standardize the masfem index (z-score)
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # Binary gender indicator
    df['gender_female'] = df['gender_mf'].astype(int)

    # Dependent variable: log-transformed fatalities (handle zeros)
    df['log_deaths'] = np.log(df['alldeaths'] + 1)

    # Control: log transformed damage (2015 dollars), use +1 to handle zeros
    df['log_ndam15'] = np.log(df['ndam15'] + 1)

    # Cast category to categorical type for modeling
    df['category'] = df['category'].astype('category')

    # Keep only columns needed for downstream modeling to reduce accidental usage of other columns
    keep_cols = [
        'masfem_z', 'gender_female', 'log_deaths', 'log_ndam15', 'category',
        'wind', 'min', 'elapsedyrs', 'year', 'alldeaths', 'ndam15', 'masfem', 'gender_mf'
    ]
    # Some datasets might not contain all extras, so intersect
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run two complementary statistical models to test whether more feminine hurricane
    names (higher masfem_z or female name) predict differences in fatalities after
    controlling for storm intensity and other covariates.

    Models fitted:
      1) OLS on log_deaths (log(alldeaths + 1)) with robust (HC3) standard errors.
      2) Negative Binomial GLM on raw alldeaths (count outcome) to respect count nature and overdispersion.

    Returns a dict with the fitted results objects: {'ols': ols_results, 'nb': nb_results}
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure the dataframe contains the needed columns
    needed = ['log_deaths', 'alldeaths', 'masfem_z', 'gender_female', 'category', 'wind', 'min', 'elapsedyrs', 'year', 'log_ndam15']
    missing = [c for c in needed if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    # OLS model on log fatalities (linearized) with categorical category control
    formula_ols = 'log_deaths ~ masfem_z + gender_female + C(category) + wind + min + elapsedyrs + year + log_ndam15'
    ols_model = smf.ols(formula_ols, data=df).fit(cov_type='HC3')

    # Negative binomial GLM on counts (alldeaths). Use the same predictors but without log outcome.
    # NB can handle overdispersion relative to Poisson.
    formula_nb = 'alldeaths ~ masfem_z + gender_female + C(category) + wind + min + elapsedyrs + year + log_ndam15'
    try:
        nb_model = smf.glm(formula_nb, data=df, family=sm.families.NegativeBinomial()).fit()
    except Exception:
        # Fallback: fit Poisson with robust SEs if NB fails to converge.
        nb_model = smf.glm(formula_nb, data=df, family=sm.families.Poisson()).fit(cov_type='HC3')

    # For interpretability: print short summaries (users can inspect full objects returned)
    print('OLS on log_deaths (HC3 SEs):')
    print(ols_model.summary())
    print('\nNegative Binomial (or Poisson fallback) on alldeaths:')
    print(nb_model.summary())

    results = {
        'ols': ols_model,
        'nb': nb_model
    }
    return results


