from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/mortgage/anonymize_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a modelling dataframe.

    Creates the following columns used in the model:
      - Approved: binary outcome (1 = accepted, 0 = denied). Uses feature14 when present, otherwise uses 1 - feature11.
      - Female: binary indicator from feature2 (1 = female, 0 = male).
      - Black, SelfEmployed, Married, BadCredit: binary controls from feature3, feature5, feature6, feature9.
      - ApplicantIncome, HousingExpenseRatio, MortgageScore, ConsumerScore, DebtToIncome, LTV: copied from feature1,4,7,8,10,12.
      - Standardized versions of continuous controls with suffix _z for numerical stability and interpretability.

    Rows with missing values in any model variable are dropped.
    """

    # Copy original dataframe to avoid modifying caller's object
    df = df.copy()

    # Create Approved outcome: prefer feature14 (1=accepted) if present, otherwise infer from feature11 (1=denied)
    # Ensure numeric
    for col in ['feature11', 'feature14']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Approved'] = None
    if 'feature14' in df.columns:
        # Use feature14 when available (1 accepted, 0 denied)
        df['Approved'] = df['feature14']
    # If feature14 is missing for some rows but feature11 is present, set Approved = 1 - feature11
    if 'feature11' in df.columns:
        mask_missing = df['Approved'].isna() if 'Approved' in df.columns else pd.Series(True, index=df.index)
        df.loc[mask_missing, 'Approved'] = 1 - df.loc[mask_missing, 'feature11']

    # Ensure binary ints
    df['Approved'] = pd.to_numeric(df['Approved'], errors='coerce')
    df.loc[df['Approved'].notnull(), 'Approved'] = df.loc[df['Approved'].notnull(), 'Approved'].astype(int)

    # Binary indicators and controls (copy / coerce)
    df['Female'] = pd.to_numeric(df.get('feature2'), errors='coerce')
    df['Black'] = pd.to_numeric(df.get('feature3'), errors='coerce')
    df['HousingExpenseRatio'] = pd.to_numeric(df.get('feature4'), errors='coerce')
    df['SelfEmployed'] = pd.to_numeric(df.get('feature5'), errors='coerce')
    df['Married'] = pd.to_numeric(df.get('feature6'), errors='coerce')
    df['MortgageScore'] = pd.to_numeric(df.get('feature7'), errors='coerce')
    df['ConsumerScore'] = pd.to_numeric(df.get('feature8'), errors='coerce')
    df['BadCredit'] = pd.to_numeric(df.get('feature9'), errors='coerce')
    df['DebtToIncome'] = pd.to_numeric(df.get('feature10'), errors='coerce')
    df['ApplicantIncome'] = pd.to_numeric(df.get('feature1'), errors='coerce')
    df['LTV'] = pd.to_numeric(df.get('feature12'), errors='coerce')

    # List of model variable columns we require
    model_vars = [
        'Approved', 'Female', 'Black', 'SelfEmployed', 'Married', 'BadCredit',
        'ApplicantIncome', 'HousingExpenseRatio', 'DebtToIncome', 'LTV',
        'MortgageScore', 'ConsumerScore'
    ]

    # Drop rows with missing values in any of model_vars
    df = df.dropna(subset=model_vars)

    # Standardize continuous variables for stability / interpretability
    cont_vars = ['ApplicantIncome', 'HousingExpenseRatio', 'DebtToIncome', 'LTV', 'MortgageScore', 'ConsumerScore']
    for col in cont_vars:
        # compute mean/std on the available rows
        mean = df[col].mean()
        std = df[col].std()
        # if std is zero (constant column), create zero column to avoid division by zero
        if pd.isna(std) or std == 0:
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Ensure binary columns are integer 0/1
    for bcol in ['Female', 'Black', 'SelfEmployed', 'Married', 'BadCredit']:
        df[bcol] = pd.to_numeric(df[bcol], errors='coerce')
        # If non-binary values are present, map truthy>0.5 to 1, else 0
        df.loc[df[bcol].notnull(), bcol] = (df.loc[df[bcol].notnull(), bcol] > 0.5).astype(int)

    # Final column list used in modeling (keeps order predictable)
    final_cols = [
        'Approved', 'Female', 'Black', 'SelfEmployed', 'Married', 'BadCredit',
        'ApplicantIncome_z', 'HousingExpenseRatio_z', 'DebtToIncome_z', 'LTV_z',
        'MortgageScore_z', 'ConsumerScore_z'
    ]
    # Make sure columns exist (they should, after transformations)
    for c in final_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # Return the transformed dataframe (with many original columns preserved plus new ones)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression model predicting Approved from Female and controls.

    Model specification (logit):
      Approved ~ Female + Black + SelfEmployed + Married + BadCredit
                 + ApplicantIncome_z + HousingExpenseRatio_z + DebtToIncome_z + LTV_z
                 + MortgageScore_z + ConsumerScore_z

    Returns a dictionary containing the fitted result object, odds ratios, confidence intervals, and a text summary.
    """

    # Prepare design matrix X and outcome y
    required = [
        'Approved', 'Female', 'Black', 'SelfEmployed', 'Married', 'BadCredit',
        'ApplicantIncome_z', 'HousingExpenseRatio_z', 'DebtToIncome_z', 'LTV_z',
        'MortgageScore_z', 'ConsumerScore_z'
    ]
    # Ensure required columns are present
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Drop any remaining rows with NA in the required columns
    df_model = df.dropna(subset=required).copy()

    y = df_model['Approved'].astype(int)
    X = df_model[[
        'Female', 'Black', 'SelfEmployed', 'Married', 'BadCredit',
        'ApplicantIncome_z', 'HousingExpenseRatio_z', 'DebtToIncome_z', 'LTV_z',
        'MortgageScore_z', 'ConsumerScore_z'
    ]]
    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (maximum likelihood)
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    # Compute odds ratios and confidence intervals
    params = result.params
    conf = result.conf_int()
    odds_ratios = np.exp(params)
    conf_odds = np.exp(conf)

    out = {
        'result': result,
        'odds_ratios': odds_ratios,
        'conf_int_odds': conf_odds,
        'summary_text': result.summary().as_text()
    }
    return out


