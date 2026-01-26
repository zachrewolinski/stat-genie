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
    """
    Transform raw hurricane dataframe for modeling:
      - Ensure numeric columns are numeric
      - Drop rows missing core variables
      - Create log_alldeaths = log(alldeaths + 1)
      - Standardize masfem to masfem_z
      - Center year to year_c
      - Create source dummies with prefix 'source_' (drop_first=True to avoid perfect collinearity)

    Returns transformed dataframe containing all columns referenced in the model code.
    """
    df = df.copy()

    # Ensure numeric conversions for key columns (coerce invalids to NaN)
    numeric_cols = ['masfem', 'masfem_mturk', 'min', 'wind', 'category', 'alldeaths', 'ndam', 'ndam15', 'elapsedyrs', 'year', 'gender_mf']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing critical variables required for primary modeling
    required = ['masfem', 'alldeaths', 'wind', 'min', 'category', 'year', 'elapsedyrs']
    df = df.dropna(subset=required)

    # Create log outcome used in OLS robustness
    df['log_alldeaths'] = np.log(df['alldeaths'] + 1)

    # Standardize masfem (z-score) for interpretability
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # Center year to reduce collinearity with intercept
    df['year_c'] = df['year'] - df['year'].mean()

    # Create categorical dummies for 'source' if present; drop one level to avoid multicollinearity
    if 'source' in df.columns:
        # Convert to string to avoid problems with categories that are not hashable / weird types
        df['source'] = df['source'].astype(str).fillna('missing')
        source_dummies = pd.get_dummies(df['source'], prefix='source', dummy_na=False)
        # drop the first dummy to serve as baseline
        if source_dummies.shape[1] > 1:
            source_dummies = source_dummies.iloc[:, 1:]
        df = pd.concat([df.reset_index(drop=True), source_dummies.reset_index(drop=True)], axis=1)

    # Ensure gender_mf exists and is numeric (0/1)
    if 'gender_mf' in df.columns:
        df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce').fillna(0).astype(int)

    # Final safety: remove any rows still with missing values in columns we'll use in the model
    # We'll identify source dummy columns by prefix 'source_' if present
    model_cols = ['masfem_z', 'alldeaths', 'log_alldeaths', 'wind', 'min', 'category', 'year_c', 'elapsedyrs', 'gender_mf']
    model_cols += [c for c in df.columns if isinstance(c, str) and c.startswith('source_')]
    df = df.dropna(subset=model_cols)

    # Return the transformed dataframe with required columns
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit two complementary models testing whether storms with more feminine names are associated with higher fatalities (proxy for fewer precautions):
      1) Primary: Negative Binomial GLM on raw count alldeaths (accounts for count nature and overdispersion)
      2) Robustness: OLS on log_alldeaths (log(count + 1)) with robust standard errors

    Returns a dict with model fit objects and prints summaries.
    """
    # Copy to avoid modifying original
    df = df.copy()

    # Build design matrix
    # Identify source dummy columns if present
    source_dummy_cols = [c for c in df.columns if c.startswith('source_')]

    exog_cols = ['masfem_z', 'wind', 'min', 'category', 'elapsedyrs', 'year_c', 'gender_mf'] + source_dummy_cols

    # Ensure columns exist
    exog_cols = [c for c in exog_cols if c in df.columns]

    X = df[exog_cols]
    X = sm.add_constant(X, has_constant='add')

    # Endogenous variables
    y_counts = df['alldeaths']
    y_log = df['log_alldeaths']

    results = {}

    # Primary model: Negative Binomial GLM with log link
    try:
        nb_model = sm.GLM(y_counts, X, family=sm.families.NegativeBinomial(link=sm.families.links.log()))
        nb_res = nb_model.fit(cov_type='HC3')
        print('\n=== Negative Binomial GLM (alldeaths) ===')
        print(nb_res.summary())
        results['neg_binom'] = nb_res
    except Exception as e:
        # If the GLM NegativeBinomial fails (sometimes due to small sample issues), capture the error
        print('Negative Binomial GLM failed:', e)
        results['neg_binom_error'] = str(e)

    # Robustness: OLS on log(alldeaths + 1)
    try:
        ols_model = sm.OLS(y_log, X)
        ols_res = ols_model.fit(cov_type='HC3')
        print('\n=== OLS on log(alldeaths + 1) ===')
        print(ols_res.summary())
        results['ols_log'] = ols_res
    except Exception as e:
        print('OLS on log outcome failed:', e)
        results['ols_error'] = str(e)

    # Return results dictionary containing fitted models (or errors)
    return results


