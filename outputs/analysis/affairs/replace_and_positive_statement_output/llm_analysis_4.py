from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/replace_and_positive_statement_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the Fair (1978) affairs dataset for modeling.

    Transformations performed:
    - Drop rows missing core variables (affairs, children).
    - Create binary outcome 'affair_binary' indicating any affair (affairs > 0).
    - Convert 'children' to binary 'children_yes' (1=yes, 0=no).
    - Convert 'gender' to binary 'gender_male' (1=male, 0=female).
    - Ensure numeric controls exist, impute medians for any remaining missings among continuous controls.
    - Standardize continuous controls into z-scores: age_z, yearsmarried_z, religiousness_z, education_z, occupation_z, rating_z.

    Returns the dataframe containing at minimum the columns referenced in the conceptual model.
    """
    df = df.copy()

    # Core required variables
    required = ['affairs', 'children']
    df = df.dropna(subset=required)

    # Ensure 'affairs' numeric
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')

    # Binary indicator: any affair
    df['affair_binary'] = (df['affairs'] > 0).astype(int)

    # children -> children_yes (1/0)
    df['children_yes'] = df['children'].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

    # gender -> gender_male (1=male, 0=female)
    if 'gender' in df.columns:
        df['gender_male'] = df['gender'].astype(str).str.strip().str.lower().map({'male': 1, 'female': 0})
    else:
        # If gender not present, create NaNs so subsequent drop will remove problematic rows
        df['gender_male'] = np.nan

    # Continuous controls to prepare and standardize
    cont_cols = [c for c in ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating'] if c in df.columns]

    # Convert to numeric and impute median for missing continuous controls
    for c in cont_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        median_val = df[c].median()
        df[c] = df[c].fillna(median_val)

    # Create standardized (z) versions of continuous controls
    from sklearn.preprocessing import StandardScaler
    if len(cont_cols) > 0:
        scaler = StandardScaler()
        z_cols = [c + '_z' for c in cont_cols]
        df[z_cols] = scaler.fit_transform(df[cont_cols])
    else:
        # if none exist, create empty columns to avoid key errors later
        for c in ['age_z', 'yearsmarried_z', 'religiousness_z', 'education_z', 'occupation_z', 'rating_z']:
            df[c] = np.nan

    # Drop rows that still have missing values in the key derived indicators we rely on
    df = df.dropna(subset=['children_yes', 'gender_male', 'affairs'])

    # Final dataframe will contain at least these columns used in modeling
    keep_cols = [
        'affairs', 'affair_binary', 'children_yes', 'gender_male',
        'age_z', 'yearsmarried_z', 'religiousness_z', 'education_z', 'occupation_z', 'rating_z'
    ]

    # Ensure all keep_cols exist in df (if some z cols were not created due to missing original cols, create NaN)
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Runs two complementary models to assess whether having children reduces engagement in extramarital affairs:
      1) Logistic regression predicting any affair (affair_binary) -- easy-to-interpret test of reduction in probability of any affair.
      2) Zero-Inflated Negative Binomial (ZINB) on the count 'affairs' to account for overdispersion and excess zeros.

    Both models include the same set of controls: gender (also treated as a moderator separately in interpretation), age_z, yearsmarried_z, religiousness_z, education_z, occupation_z, rating_z.

    Returns a dict with model result objects: {'logit': logit_result, 'zinb': zinb_result}.
    """
    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

    df = df.copy()

    # Prepare regressors
    control_cols = ['gender_male', 'age_z', 'yearsmarried_z', 'religiousness_z', 'education_z', 'occupation_z', 'rating_z']
    exog_cols = ['children_yes'] + control_cols

    # Ensure exog columns exist
    missing = [c for c in exog_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Add constant
    X = df[exog_cols]
    X = sm.add_constant(X, has_constant='add')

    # 1) Logistic regression for any affair
    y_logit = df['affair_binary']
    logit_model = sm.Logit(y_logit, X).fit(disp=False)

    # 2) Zero-inflated negative binomial for counts (accounts for many zeros and overdispersion)
    y_count = df['affairs']

    # Use same exog for both count and inflation parts; inflation modeled with a logit (probability of an 'excess' zero)
    zinb_mod = ZeroInflatedNegativeBinomialP(endog=y_count, exog=X, exog_infl=X, inflation='logit')
    zinb_res = zinb_mod.fit(disp=False, maxiter=200)

    # Return both result objects for further inspection
    return {
        'logit': logit_model,
        'zinb': zinb_res
    }


