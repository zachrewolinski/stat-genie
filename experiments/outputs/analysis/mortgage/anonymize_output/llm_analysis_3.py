from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/anonymize_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a modeling-ready dataframe. The function:
    - Makes a copy of the input
    - Drops rows missing the outcome (feature14) or the main IV (feature2)
    - Creates interpretable column names
    - Drops rows with missing values in any variable used in the model
    - Standardizes continuous variables (z-scoring)
    - Creates an interaction term Female_Black for moderation tests
    Returns the transformed dataframe containing exactly the columns referenced in the conceptual variables.
    """
    df = df.copy()

    # REQUIRED RAW COLUMNS (per provided schema):
    # feature2 -> female indicator
    # feature14 -> accepted indicator (1 = accepted)
    # Map and drop rows where these are missing
    df = df.dropna(subset=['feature2', 'feature14'])

    # Create primary variables with explicit names
    df['Female'] = df['feature2'].astype(int)
    # Use feature14 as Accepted (schema: 1 accepted, 0 denied)
    df['Accepted'] = df['feature14'].astype(int)

    # Race (Black indicator)
    if 'feature3' in df.columns:
        df['Black'] = df['feature3'].astype(int)
    else:
        # if missing, create a zero column (but prefer raising); here we create zeros to keep pipeline robust
        df['Black'] = 0

    # Binary controls
    df['SelfEmployed'] = df['feature5'].fillna(0).astype(int)
    df['Married'] = df['feature6'].fillna(0).astype(int)
    df['BadCredit'] = df['feature9'].fillna(0).astype(int)
    df['PMI_Denied'] = df['feature13'].fillna(0).astype(int)

    # Continuous variables (create copies; missing values handled below)
    df['LoanAmount'] = df['feature1']
    df['HousingExpenseRatio'] = df['feature4']
    df['MortgageScore'] = df['feature7']
    df['ConsumerScore'] = df['feature8']
    df['DebtToIncome'] = df['feature10']
    df['LoanToValue'] = df['feature12']

    # Select columns that will be used and drop rows with any missing values among them
    model_cols = [
        'Accepted', 'Female', 'Black', 'SelfEmployed', 'Married', 'BadCredit', 'PMI_Denied',
        'LoanAmount', 'HousingExpenseRatio', 'MortgageScore', 'ConsumerScore', 'DebtToIncome', 'LoanToValue'
    ]
    df = df.dropna(subset=model_cols)

    # Standardize continuous variables (z-scoring) for stable estimation
    cont_cols = ['LoanAmount', 'HousingExpenseRatio', 'MortgageScore', 'ConsumerScore', 'DebtToIncome', 'LoanToValue']
    # compute mean/std and create _z columns
    for col in cont_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0 or np.isnan(std):
            # avoid division by zero; if std=0 create zeros
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Interaction term for moderation test
    df['Female_Black'] = df['Female'] * df['Black']

    # Final column list to keep (only columns referenced in model and conceptual variables)
    final_cols = [
        'Accepted', 'Female', 'Black', 'Female_Black',
        'LoanAmount_z', 'HousingExpenseRatio_z', 'MortgageScore_z', 'ConsumerScore_z', 'DebtToIncome_z', 'LoanToValue_z',
        'SelfEmployed', 'Married', 'BadCredit', 'PMI_Denied'
    ]

    # Keep only final columns (if any are missing because original features absent, function will raise)
    missing_final = [c for c in final_cols if c not in df.columns]
    if missing_final:
        raise ValueError(f"Missing required transformed columns: {missing_final}")

    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression (binomial GLM) predicting Accepted from Female,
    controls, and a Female x Black interaction to test whether the gender effect
    differs by race. Returns the fitted GLMResults object and average marginal effects.
    """
    # Ensure inputs exist
    required_cols = [
        'Accepted', 'Female', 'Black', 'Female_Black',
        'LoanAmount_z', 'HousingExpenseRatio_z', 'MortgageScore_z', 'ConsumerScore_z', 'DebtToIncome_z', 'LoanToValue_z',
        'SelfEmployed', 'Married', 'BadCredit', 'PMI_Denied'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe for modeling: {missing}")

    # Prepare design matrix
    X_cols = [
        'Female', 'Black', 'Female_Black',
        'LoanAmount_z', 'HousingExpenseRatio_z', 'MortgageScore_z', 'ConsumerScore_z', 'DebtToIncome_z', 'LoanToValue_z',
        'SelfEmployed', 'Married', 'BadCredit', 'PMI_Denied'
    ]
    X = df[X_cols].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['Accepted'].astype(int)

    # Fit binomial GLM with logit link (equivalent to logistic regression)
    model_glm = sm.GLM(y, X, family=sm.families.Binomial())
    results = model_glm.fit()

    # Compute average marginal effects (AME) for easier interpretation of the binary predictor 'Female'
    try:
        marg_eff = results.get_margeff(at='overall', method='dydx')
    except Exception:
        marg_eff = None

    # Return both the fitted results and marginal effects (if computed)
    return {'results': results, 'marginal_effects': marg_eff}


