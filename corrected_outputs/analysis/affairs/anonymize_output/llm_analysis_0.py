from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/affairs/anonymize_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Fair (Psychology Today) survey data into analysis-ready dataframe.

    Produces the following columns used in the model:
      - RespondentID: from feature1 (kept for traceability)
      - AffairCount: from feature2 (count proxy; uses original coded values)
      - AnyAffair: binary indicator AffairCount > 0 (useful for diagnostics)
      - Children: binary indicator 1 if children in the marriage (feature6 == 'yes'), 0 otherwise
      - Male: binary gender indicator (1 = male, 0 = female)
      - Children_Male: interaction term Children * Male
      - Age, YearsMarried, Religiosity, Education, Occupation, MaritalHappiness: mapped from feature4..feature10

    Rows with missing values in any of the model variables are dropped.
    """
    # Work on a copy
    df = df.copy()

    # Rename / create ID
    if 'feature1' in df.columns:
        df['RespondentID'] = df['feature1']
    else:
        df['RespondentID'] = np.arange(len(df)) + 1

    # Affair count (use original coding as numeric proxy)
    df['AffairCount'] = pd.to_numeric(df['feature2'], errors='coerce')
    # Also create a binary indicator of any affair for diagnostics
    df['AnyAffair'] = (df['AffairCount'] > 0).astype(int)

    # Children: feature6 expected to be categorical 'yes'/'no'
    # Be robust to capitalization and missing values
    df['Children'] = df['feature6'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # Gender: feature3 expected to be 'male'/'female'
    df['Male'] = df['feature3'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})

    # Interaction (moderator) term: Children x Male
    # If either is NA, result will be NA and those rows are dropped later
    df['Children_Male'] = df[['Children', 'Male']].prod(axis=1)

    # Controls: numeric conversions; coerce errors to NaN
    df['Age'] = pd.to_numeric(df['feature4'], errors='coerce')
    df['YearsMarried'] = pd.to_numeric(df['feature5'], errors='coerce')
    df['Religiosity'] = pd.to_numeric(df['feature7'], errors='coerce')
    df['Education'] = pd.to_numeric(df['feature8'], errors='coerce')
    df['Occupation'] = pd.to_numeric(df['feature9'], errors='coerce')
    df['MaritalHappiness'] = pd.to_numeric(df['feature10'], errors='coerce')

    # Select the columns we will actually use and drop rows with missing values in any of them
    model_cols = [
        'RespondentID', 'AffairCount', 'AnyAffair', 'Children', 'Male', 'Children_Male',
        'Age', 'YearsMarried', 'Religiosity', 'Education', 'Occupation', 'MaritalHappiness'
    ]

    # Ensure all needed columns exist
    for col in model_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Drop rows with missing values in model columns
    df = df.dropna(subset=[c for c in model_cols if c not in ['RespondentID']])

    # Ensure numeric dtypes for modeling
    numeric_cols = ['AffairCount', 'AnyAffair', 'Children', 'Male', 'Children_Male',
                    'Age', 'YearsMarried', 'Religiosity', 'Education', 'Occupation', 'MaritalHappiness']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Final drop in case coercion introduced NaNs
    df = df.dropna(subset=numeric_cols)

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a zero-inflated negative binomial model predicting AffairCount from Children
    (main IV), controlling for demographic and relationship factors, and including
    an interaction term between Children and Male to test gender moderation.

    Returns the fitted statsmodels results object for the count model.
    """
    # Import the needed class locally (statsmodels may not have it in top-level)
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

    # Prepare endogenous and exogenous matrices
    endog = df['AffairCount'].astype(float)

    exog_vars = ['Children', 'Male', 'Children_Male', 'Age', 'YearsMarried',
                 'Religiosity', 'Education', 'Occupation', 'MaritalHappiness']
    exog = sm.add_constant(df[exog_vars], has_constant='add')

    # For the zero-inflation (logit) part use a smaller set of predictors that plausibly
    # predict the structural zero (no opportunity / no propensity): Children, Male,
    # YearsMarried, Religiosity. Include constant.
    exog_infl = sm.add_constant(df[['Children', 'Male', 'YearsMarried', 'Religiosity']], has_constant='add')

    # Fit Zero-Inflated Negative Binomial (parameterization p=1). If convergence issues occur
    # increase maxiter or switch method.
    zinb = ZeroInflatedNegativeBinomialP(endog, exog, exog_infl=exog_infl, p=1)
    try:
        results = zinb.fit(maxiter=100, method='newton', disp=False)
    except Exception:
        # fallback: allow more iterations and different method
        results = zinb.fit(maxiter=500, method='bfgs', disp=False)

    # For convenience return the fitted results object (has .summary())
    return results


