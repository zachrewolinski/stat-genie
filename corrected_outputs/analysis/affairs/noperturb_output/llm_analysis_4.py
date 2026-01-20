from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/affairs/noperturb_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Columns required for analysis
    required = ['affairs', 'children', 'gender', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required)

    # Encode children: expected values are 'yes'/'no' (factor). Create binary column 'Children'
    df['Children'] = df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Encode gender: create Male = 1 if male, 0 if female
    df['Male'] = df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})

    # If mapping produced NaNs (unexpected levels), drop those rows
    df = df.dropna(subset=['Children', 'Male'])

    # Ensure numeric columns are numeric
    numeric_cols = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating', 'affairs']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows with newly coerced NaNs in numeric columns
    df = df.dropna(subset=numeric_cols)

    # Standardize continuous controls to z-scores for interpretability
    df['Age_z'] = (df['age'] - df['age'].mean()) / df['age'].std(ddof=0)
    df['Yearsmarried_z'] = (df['yearsmarried'] - df['yearsmarried'].mean()) / df['yearsmarried'].std(ddof=0)
    df['Religiousness_z'] = (df['religiousness'] - df['religiousness'].mean()) / df['religiousness'].std(ddof=0)
    df['Education_z'] = (df['education'] - df['education'].mean()) / df['education'].std(ddof=0)
    df['Occupation_z'] = (df['occupation'] - df['occupation'].mean()) / df['occupation'].std(ddof=0)
    df['Rating_z'] = (df['rating'] - df['rating'].mean()) / df['rating'].std(ddof=0)

    # Keep only the columns required for modeling (but keep original affairs column as DV)
    final_cols = ['affairs', 'Children', 'Male', 'Age_z', 'Yearsmarried_z', 'Religiousness_z', 'Education_z', 'Occupation_z', 'Rating_z']
    df = df[final_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    # Model the association between having children and frequency of extramarital affairs,
    # controlling for demographic and marriage-related covariates.
    # We use a Zero-Inflated Negative Binomial (ZINB) model because 'affairs' is a nonnegative,
    # overdispersed count-like variable with many zeros.

    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

    # Ensure a copy
    df = df.copy()

    # Define model covariates (count model part)
    exog_vars = ['Children', 'Male', 'Age_z', 'Yearsmarried_z', 'Religiousness_z', 'Education_z', 'Occupation_z', 'Rating_z']
    exog = sm.add_constant(df[exog_vars], has_constant='add')

    # Endogenous (dependent) variable
    endog = df['affairs']

    # For the inflation (logit) part, we use a smaller set of covariates that plausibly predict structural zeros
    # (e.g., presence of children and gender). Include a constant as well.
    exog_infl = sm.add_constant(df[['Children', 'Male']], has_constant='add')

    # Fit Zero-Inflated Negative Binomial model
    zinb = ZeroInflatedNegativeBinomialP(endog, exog, exog_infl=exog_infl, inflation='logit')

    # Fit using a robust optimizer; suppress iterative output (disp=0)
    try:
        results = zinb.fit(method='bfgs', maxiter=100, disp=0)
    except Exception:
        # Fallback to default fit if bfgs fails
        results = zinb.fit(disp=0)

    # Return the fitted results object (contains parameter estimates, standard errors, summary, etc.)
    return results


