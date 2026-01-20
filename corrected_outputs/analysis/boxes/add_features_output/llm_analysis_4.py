from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/boxes/add_features_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis-ready dataframe.

    Output columns used in modeling:
      - y_mn: dependent variable for multinomial model (0=unchosen, 1=majority, 2=minority)
      - age_c: centered age
      - age_c2: squared centered age
      - culture: categorical culture/site identifier as string
      - is_boy: binary gender (1 = boy, 0 = girl)
      - majority_first: binary indicator (0/1) as provided
      - religiousness: numeric control
      - school: school identifier (kept for clustering)
      - ind: original row id (kept)

    Steps:
      - drop rows missing key variables
      - create centered age and squared term
      - create y_mn for MNLogit (0..J-1)
      - coerce types
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Key columns required for analysis
    required = ['y', 'age', 'culture', 'gender', 'majority_first', 'religiousness']
    # Drop rows with missing values in required columns
    df = df.dropna(subset=required)

    # Ensure y is integer and within expected categories
    df['y'] = df['y'].astype(int)
    # Keep only rows with valid outcome categories 1,2,3
    df = df[df['y'].isin([1, 2, 3])].copy()

    # Create y_mn: 0-based outcome for MNLogit (0=unchosen, 1=majority, 2=minority)
    df['y_mn'] = df['y'].astype(int) - 1

    # Age: numeric; center around sample mean to improve interpretability
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['age'])
    age_mean = df['age'].mean()
    df['age_c'] = df['age'] - age_mean
    df['age_c2'] = df['age_c'] ** 2

    # Culture: treat as categorical string for later get_dummies
    df['culture'] = df['culture'].astype(int).astype(str)

    # Gender: create is_boy binary (1 if boy (gender==2), 0 if girl (gender==1)). If other coding present, fallback to nan and drop.
    df['gender'] = pd.to_numeric(df['gender'], errors='coerce')
    df = df[df['gender'].isin([1, 2])]
    df['is_boy'] = (df['gender'] == 2).astype(int)

    # majority_first should already be 0/1; coerce to int
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce').fillna(0).astype(int)

    # religiousness numeric as given; coerce
    df['religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')

    # Keep school and ind if present; coerce to string for clustering
    if 'school' in df.columns:
        df['school'] = df['school'].astype(str)
    if 'ind' in df.columns:
        # keep original row identifier
        df['ind'] = df['ind']

    # Final drop of rows with any remaining NA in model columns
    model_cols = ['y_mn', 'age_c', 'age_c2', 'culture', 'is_boy', 'majority_first', 'religiousness']
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a multinomial logistic regression predicting children's choice (y_mn) with:
      - Main predictors: age_c, age_c2
      - Culture main effects and interactions between age terms and culture to estimate culture-specific developmental trajectories
      - Controls: is_boy, majority_first, religiousness

    Returns the fitted statsmodels MNLogit result (attempts cluster-robust SE by school if available).
    """
    # Build culture dummies (drop one reference to avoid multicollinearity)
    culture_dummies = pd.get_dummies(df['culture'], prefix='culture', drop_first=True)

    # Base covariates
    covariates = pd.DataFrame({
        'age_c': df['age_c'],
        'age_c2': df['age_c2'],
        'is_boy': df['is_boy'],
        'majority_first': df['majority_first'],
        'religiousness': df['religiousness']
    }, index=df.index)

    # Include culture main effects
    exog = pd.concat([covariates, culture_dummies], axis=1)

    # Add interactions between age terms and culture dummies so each culture can have a different slope/curvature
    for col in culture_dummies.columns:
        exog[f'{col}_x_age'] = culture_dummies[col] * df['age_c']
        exog[f'{col}_x_age2'] = culture_dummies[col] * df['age_c2']

    # Add intercept
    exog = sm.add_constant(exog, has_constant='add')

    # Endogenous variable: 0..J-1
    endog = df['y_mn'].astype(int)

    # Fit multinomial logit via statsmodels
    try:
        mnlogit = sm.MNLogit(endog, exog)
        res = mnlogit.fit(method='newton', maxiter=200, disp=False)
    except Exception:
        # Fallback to different optimizer if Newton fails
        mnlogit = sm.MNLogit(endog, exog)
        res = mnlogit.fit(method='bfgs', maxiter=200, disp=False)

    # Attempt to get cluster-robust standard errors by school if the school column is present and has >1 cluster
    clustered_result = None
    if 'school' in df.columns:
        try:
            # Only compute clustering if there are at least 2 clusters
            n_clusters = df['school'].nunique()
            if n_clusters > 1:
                clustered_result = res.get_robustcov_results(cov_type='cluster', groups=df['school'])
        except Exception:
            clustered_result = None

    # Return the clustered_result if available, otherwise return the original fit
    if clustered_result is not None:
        return clustered_result
    else:
        return res


