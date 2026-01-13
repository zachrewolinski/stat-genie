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
    Transform the raw hurricane dataframe for modeling.

    Outputs (columns included in the returned df):
      - alldeaths: original death counts (DV)
      - log_alldeaths: np.log1p(alldeaths) (robustness DV)
      - masfem_c: z-scored masfem (IV, continuous)
      - gender_mf: original binary female-name indicator (0/1)
      - wind_c: z-scored wind speed
      - category: original category (1-5); treated as categorical in model
      - min_c: z-scored minimum pressure
      - elapsedyrs_c: z-scored elapsedyrs

    Rows with missing values on required columns are dropped.
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Required raw columns for the planned analysis
    required_cols = [
        'alldeaths',  # DV
        'masfem',     # IV (continuous coder ratings)
        'gender_mf',  # binary female/male name
        'wind',       # control: wind speed
        'category',   # control: Saffir-Simpson category
        'min',        # control: minimum pressure
        'elapsedyrs'  # control: elapsed years
    ]

    # Drop rows missing any of the required fields
    df = df.dropna(subset=required_cols)

    # Ensure numeric types where appropriate
    for col in ['alldeaths', 'masfem', 'gender_mf', 'wind', 'category', 'min', 'elapsedyrs']:
        # coerce to numeric if possible
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Drop rows that became NA after coercion
    df = df.dropna(subset=required_cols)

    # Derived outcome: log(1 + deaths) for OLS robustness
    df['log_alldeaths'] = np.log1p(df['alldeaths'].astype(float))

    # Standardize continuous predictors (z-score). Use .std(ddof=0) to match population s.d.
    def zscore(series):
        s = series.astype(float)
        return (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1.0)

    df['masfem_c'] = zscore(df['masfem'])
    df['wind_c'] = zscore(df['wind'])
    df['min_c'] = zscore(df['min'])
    df['elapsedyrs_c'] = zscore(df['elapsedyrs'])

    # Keep only the columns we need for modeling plus originals for reference
    model_cols = [
        'alldeaths', 'log_alldeaths',
        'masfem_c', 'masfem', 'gender_mf',
        'wind_c', 'wind', 'category', 'min_c', 'min', 'elapsedyrs_c', 'elapsedyrs'
    ]
    # If any of these are missing from the df (unlikely), keep what we have
    model_cols = [c for c in model_cols if c in df.columns]

    df = df[model_cols].reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit the main Negative Binomial model (counts) and an OLS robustness model on log(1+deaths).

    Returns a dict with fitted model results objects:
      - 'nb_model': GLM NegativeBinomial fitted result (primary)
      - 'ols_model': OLS fitted result on log(1+deaths) (robustness)

    Both models include masfem (standardized) as the main predictor and control for
    gender_mf, storm severity (wind, category, min pressure), and elapsed years.
    Category is treated as a categorical factor in the formulas (C(category)).
    Robust (HC3) standard errors are requested for inference.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['alldeaths', 'log_alldeaths', 'masfem_c', 'gender_mf', 'wind_c', 'category', 'min_c', 'elapsedyrs_c']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f'Missing required columns for modeling: {missing}')

    # Formula: count model (Negative Binomial)
    formula_nb = 'alldeaths ~ masfem_c + gender_mf + wind_c + C(category) + min_c + elapsedyrs_c'

    # Fit GLM Negative Binomial
    nb_model = smf.glm(formula=formula_nb, data=df, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')

    # Robustness: OLS on log(1 + deaths)
    formula_ols = 'log_alldeaths ~ masfem_c + gender_mf + wind_c + C(category) + min_c + elapsedyrs_c'
    ols_model = smf.ols(formula=formula_ols, data=df).fit(cov_type='HC3')

    # Compute and attach a simple overdispersion diagnostic (variance/mean for counts)
    mean_deaths = df['alldeaths'].mean()
    var_deaths = df['alldeaths'].var()
    overdispersion = None
    if mean_deaths > 0:
        overdispersion = float(var_deaths / mean_deaths)

    results = {
        'nb_model': nb_model,
        'ols_model': ols_model,
        'overdispersion_ratio_var_over_mean': overdispersion,
        'n_observations': int(df.shape[0])
    }

    # Print brief summaries for quick inspection (optional)
    print('Negative Binomial model summary:')
    print(nb_model.summary())
    print('\nOLS (log1p deaths) robustness summary:')
    print(ols_model.summary())
    if overdispersion is not None:
        print(f'Overdispersion (var/mean) for alldeaths: {overdispersion:.3f}')

    return results


