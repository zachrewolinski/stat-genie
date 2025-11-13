from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Keep relevant columns and drop rows with missing critical values
    required = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'ndam15', 'elapsedyrs', 'category']
    df = df.dropna(subset=required)

    # Create binary female-name indicator
    # In dataset gender_mf: 1 = female, 0 = male
    df['female_name'] = df['gender_mf'].astype(int)

    # Log-transform monetary damage to reduce skew
    df['log_ndam15'] = np.log1p(df['ndam15'])

    # Log-transform deaths for OLS modeling
    df['log_alldeaths'] = np.log1p(df['alldeaths'])

    # Standardize continuous predictors (z-score). Use sample std (ddof=0) via pandas .std() default ddof=1; to get population-like z, use ddof=0 via numpy
    # We'll compute z = (x - mean)/std with ddof=0 to be explicit
    def zscore(series):
        arr = series.astype(float).to_numpy()
        mu = np.nanmean(arr)
        sigma = np.nanstd(arr)
        if sigma == 0:
            return series * 0.0
        return (series - mu) / sigma

    df['masfem_z'] = zscore(df['masfem'])
    df['wind_z'] = zscore(df['wind'])
    df['min_z'] = zscore(df['min'])
    df['log_ndam15_z'] = zscore(df['log_ndam15'])
    df['elapsedyrs_z'] = zscore(df['elapsedyrs'])

    # Convert category to integer and create dummy variables for categories 2-5 (reference: category 1)
    df['category'] = df['category'].astype(int)
    cats = pd.get_dummies(df['category'], prefix='category')
    # Ensure consistent dummy columns for categories 2-5 (use 1 as reference). If some categories are missing in this dataset, add columns with zeros so downstream model code can rely on their presence.
    for c in [2,3,4,5]:
        col = f'category_{c}'
        if col not in cats.columns:
            cats[col] = 0
    # Keep only category_2..category_5 (drop category_1 to be the reference)
    cats = cats[[f'category_{c}' for c in [2,3,4,5]]]
    df = pd.concat([df, cats], axis=1)

    # Final columns we will use in modeling (keep original alldeaths too)
    # Return the transformed dataframe (with all original columns plus newly created ones)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """Runs the primary negative-binomial count model (alldeaths) and a robustness OLS on log(1+alldeaths).

    Returns a dict with statsmodels results objects: {'nb_model': ..., 'ols_model': ...}
    """
    # Work on a copy
    df = df.copy()

    # Define predictors used in both models; ensure the expected dummy columns exist
    predictor_cols = [
        'masfem_z',
        'female_name',
        'wind_z',
        'min_z',
        'log_ndam15_z',
        'elapsedyrs_z',
        'category_2',
        'category_3',
        'category_4',
        'category_5'
    ]

    # Check presence
    missing = [c for c in predictor_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing predictor columns in transformed dataframe: {missing}")

    X = df[predictor_cols].astype(float)
    X = sm.add_constant(X)

    # Outcome for count model
    y_count = df['alldeaths'].astype(float)

    # Fit Negative Binomial GLM for counts (robust to overdispersion relative to Poisson)
    try:
        nb_model = sm.GLM(y_count, X, family=sm.families.NegativeBinomial()).fit()
    except Exception:
        # If GLM NB fails, fall back to Poisson with robust covariance
        poisson = sm.GLM(y_count, X, family=sm.families.Poisson()).fit(cov_type='HC0')
        nb_model = poisson

    # Robustness: OLS on log(1 + deaths)
    y_log = df['log_alldeaths'].astype(float)
    ols_model = sm.OLS(y_log, X).fit()

    # Return fitted model results
    return {
        'nb_model': nb_model,
        'ols_model': ols_model
    }


