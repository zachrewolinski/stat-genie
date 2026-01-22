from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/negative_leading_statement_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the Fair "affairs" dataset for regression analysis.

    Produces the following new columns used in modeling:
      - has_children: binary (1=yes, 0=no)
      - is_female: binary (1=female, 0=male)
      - children_gender_interaction: interaction term has_children * is_female
      - age_s, yearsmarried_s, religiousness_s, education_s, occupation_s, rating_s: standardized controls
      - log_affairs: log(affairs + 1) for OLS robustness check

    Drops rows with missing values in any of the variables needed for the models.
    """
    df = df.copy()

    # Columns required for analysis
    required_cols = [
        'affairs', 'children', 'gender', 'age', 'yearsmarried',
        'religiousness', 'education', 'occupation', 'rating'
    ]

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_cols)

    # Map children to binary indicator
    df['has_children'] = df['children'].astype(str).str.lower().map({'yes': 1, 'no': 0})
    # If mapping produced NaNs for unexpected values, drop them
    df = df[df['has_children'].isin([0, 1])]

    # Map gender to binary female indicator
    df['is_female'] = df['gender'].astype(str).str.lower().map({'female': 1, 'male': 0})
    df = df[df['is_female'].isin([0, 1])]

    # Interaction term for moderation test
    df['children_gender_interaction'] = df['has_children'] * df['is_female']

    # Standardize continuous / ordinal covariates (mean 0, sd 1) for interpretability
    def standardize(series: pd.Series, name: str) -> pd.Series:
        mean = series.mean()
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            # If no variation, return zeros
            return pd.Series(0.0, index=series.index)
        return (series - mean) / std

    df['age_s'] = standardize(df['age'].astype(float), 'age')
    df['yearsmarried_s'] = standardize(df['yearsmarried'].astype(float), 'yearsmarried')
    df['religiousness_s'] = standardize(df['religiousness'].astype(float), 'religiousness')
    df['education_s'] = standardize(df['education'].astype(float), 'education')
    df['occupation_s'] = standardize(df['occupation'].astype(float), 'occupation')
    df['rating_s'] = standardize(df['rating'].astype(float), 'rating')

    # Create a log-transformed affairs measure for OLS robustness
    df['log_affairs'] = np.log1p(df['affairs'].astype(float))

    # Keep only columns needed for downstream modeling to reduce accidental usage of others
    keep_cols = [
        'affairs', 'log_affairs', 'has_children', 'is_female', 'children_gender_interaction',
        'age_s', 'yearsmarried_s', 'religiousness_s', 'education_s', 'occupation_s', 'rating_s'
    ]

    # Return the reduced dataframe (but keep original columns as well if you prefer by removing the following line)
    return df[keep_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit primary and robustness models to estimate the effect of having children on reported extramarital affairs.

    Primary model: Negative Binomial regression (counts model) of affairs on has_children,
                   controlling for covariates and testing moderation by gender via interaction term.

    Robustness model: OLS on log(affairs + 1) with same covariates.

    Returns a dictionary with fitted model result objects and summary strings.
    """
    results = {}

    # Ensure required columns exist
    needed = [
        'affairs', 'has_children', 'is_female', 'children_gender_interaction',
        'age_s', 'yearsmarried_s', 'religiousness_s', 'education_s', 'occupation_s', 'rating_s',
        'log_affairs'
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Build design matrix
    exog_vars = [
        'has_children', 'is_female', 'children_gender_interaction',
        'age_s', 'yearsmarried_s', 'religiousness_s', 'education_s', 'occupation_s', 'rating_s'
    ]
    X = df[exog_vars]
    X = sm.add_constant(X, has_constant='add')
    y = df['affairs'].astype(float)

    # 1) Negative Binomial GLM (primary count model)
    try:
        nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
        nb_res = nb_model.fit(maxiter=100, disp=False)
        results['neg_binom'] = nb_res
        results['neg_binom_summary'] = nb_res.summary().as_text()
    except Exception as e:
        # If negative binomial fails, capture error
        results['neg_binom_error'] = str(e)

    # 2) OLS on log(affairs + 1) as robustness
    try:
        y_log = df['log_affairs'].astype(float)
        ols_model = sm.OLS(y_log, X)
        ols_res = ols_model.fit()
        results['ols_log'] = ols_res
        results['ols_log_summary'] = ols_res.summary().as_text()
    except Exception as e:
        results['ols_log_error'] = str(e)

    # 3) Simple mean comparison for raw descriptive check (mean affairs by children)
    try:
        means = df.groupby('has_children')['affairs'].agg(['mean', 'count', 'std']).rename(columns={'mean': 'affairs_mean'})
        results['means_by_children'] = means.reset_index().to_dict(orient='list')
    except Exception as e:
        results['means_error'] = str(e)

    # Return all results; callers can inspect the fitted model objects and summary strings
    return results


