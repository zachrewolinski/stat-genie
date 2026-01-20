from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/amtl/replace_with_rvs_output/amtl.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for binomial (AMTL) modeling.

    Transformations performed:
    - Drop rows missing essential columns
    - Remove rows with invalid socket or missing-tooth counts
    - Create IsHuman indicator (Homo sapiens == 1)
    - Mean-center age and prob_male for numerical stability
    - Create tooth-class dummy variables with 'Anterior' as the reference

    The returned dataframe includes the columns used in the model:
    ['num_amtl','sockets','IsHuman','age_c','prob_male_c','tooth_class_Posterior','tooth_class_Premolar','genus']
    """
    df = df.copy()

    # Drop rows missing key fields required for modeling
    df = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class'])

    # Keep only rows with valid socket counts and valid AMTL counts
    df = df[df['sockets'] > 0]
    df = df[(df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])]

    # Create binary indicator for modern humans (Homo sapiens)
    # Use string match to be robust to types
    df['genus'] = df['genus'].astype(str)
    df['IsHuman'] = (df['genus'].str.strip() == 'Homo sapiens').astype(int)

    # Center continuous covariates for numerical stability
    df['age_c'] = df['age'] - df['age'].mean()
    df['prob_male_c'] = df['prob_male'] - df['prob_male'].mean()

    # Create tooth-class dummy variables with 'Anterior' as the reference level
    tooth_dummies = pd.get_dummies(df['tooth_class'].astype(str), prefix='tooth_class')
    # Ensure both expected dummy columns exist (if a level is absent in the sample, add column of zeros)
    for col in ['tooth_class_Posterior', 'tooth_class_Premolar']:
        if col not in tooth_dummies.columns:
            tooth_dummies[col] = 0
    df = pd.concat([df, tooth_dummies[['tooth_class_Posterior', 'tooth_class_Premolar']]], axis=1)

    # Final sanity checks (optional): ensure integer counts
    df['num_amtl'] = df['num_amtl'].astype(int)
    df['sockets'] = df['sockets'].astype(int)

    # Keep only columns necessary for modeling (but retain genus for diagnostics)
    # Note: we return the full dataframe including derived columns; model() will select columns it needs
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM for AMTL frequency.

    Model specification (primary):
      endog: (num_amtl, sockets - num_amtl) as a two-column binomial endog
      exog: constant + IsHuman + age_c + prob_male_c + tooth_class_Posterior + tooth_class_Premolar

    Interpretation: the coefficient for IsHuman tests whether modern humans have higher (positive coef)
    or lower (negative coef) odds of tooth loss compared to the reference (non-human) after adjusting
    for age, sex, and tooth class.

    Returns the fitted statsmodels GLMResultsWrapper object.
    """
    # Required model columns
    X_cols = ['IsHuman', 'age_c', 'prob_male_c', 'tooth_class_Posterior', 'tooth_class_Premolar']

    # Check that required columns are present
    missing = [c for c in X_cols + ['num_amtl', 'sockets'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Design matrix with intercept
    X = sm.add_constant(df[X_cols], has_constant='add')

    # Construct binomial endog as (successes, failures)
    # statsmodels accepts an (n,2) array where columns are successes and failures
    endog = np.vstack([df['num_amtl'].values, (df['sockets'] - df['num_amtl']).values]).T

    # Fit binomial GLM (logit link by default)
    glm_binom = sm.GLM(endog, X, family=sm.families.Binomial())
    results = glm_binom.fit()

    # Return the fitted results object for inspection (summary, params, conf_int, etc.)
    return results


