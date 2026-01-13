from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/noperturb_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Drop index-like column if present
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Rename columns to canonical names used in modeling
    rename_map = {
        'female': 'Female',
        'black': 'Black',
        'housing_expense_ratio': 'HousingExpenseRatio',
        'self_employed': 'SelfEmployed',
        'married': 'Married',
        'mortgage_credit': 'MortgageCredit',
        'consumer_credit': 'ConsumerCredit',
        'bad_history': 'BadHistory',
        'PI_ratio': 'PI_ratio',
        'loan_to_value': 'LoanToValue',
        'denied_PMI': 'Denied_PMI',
        'accept': 'Accept',
        'deny': 'Deny'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Keep rows with observed outcome and gender
    df = df.dropna(subset=['Accept', 'Female'])

    # Ensure key binary fields are numeric
    for col in ['Female', 'Black', 'SelfEmployed', 'Married', 'BadHistory', 'Accept']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing core financial predictors used as controls
    required_financial = [c for c in ['MortgageCredit', 'ConsumerCredit', 'PI_ratio', 'LoanToValue'] if c in df.columns]
    if len(required_financial) > 0:
        df = df.dropna(subset=required_financial)

    # Create standardized (z-scored) versions of the two credit measures and a composite credit index
    if 'MortgageCredit' in df.columns:
        df['MortgageCredit_z'] = (df['MortgageCredit'] - df['MortgageCredit'].mean()) / (df['MortgageCredit'].std(ddof=0) if df['MortgageCredit'].std(ddof=0) != 0 else 1)
    else:
        df['MortgageCredit_z'] = 0.0

    if 'ConsumerCredit' in df.columns:
        df['ConsumerCredit_z'] = (df['ConsumerCredit'] - df['ConsumerCredit'].mean()) / (df['ConsumerCredit'].std(ddof=0) if df['ConsumerCredit'].std(ddof=0) != 0 else 1)
    else:
        df['ConsumerCredit_z'] = 0.0

    df['CreditComposite'] = df[['MortgageCredit_z', 'ConsumerCredit_z']].mean(axis=1)

    # Ensure binary columns are integers (0/1)
    for col in ['Female', 'Black', 'SelfEmployed', 'Married', 'BadHistory', 'Accept']:
        if col in df.columns:
            # coerce to 0/1 integer where possible
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Select and return the final dataframe with the exact column names used in the model
    final_cols = [
        'Accept', 'Female', 'Black', 'MortgageCredit', 'ConsumerCredit', 'CreditComposite',
        'BadHistory', 'PI_ratio', 'LoanToValue', 'HousingExpenseRatio', 'SelfEmployed', 'Married'
    ]

    # Keep only columns that actually exist (some datasets might be missing optional columns)
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols].copy()

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Work on a copy
    df = df.copy()

    # Define outcome and regressors. Use CreditComposite (standardized composite) plus standard controls.
    y = df['Accept']

    feature_cols = ['Female', 'Black', 'CreditComposite', 'BadHistory', 'PI_ratio', 'LoanToValue', 'HousingExpenseRatio', 'SelfEmployed', 'Married']
    # Keep only the features that exist in the dataframe (transform ensures most exist)
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    X = sm.add_constant(X, has_constant='add')

    # Fit a logistic regression (logit) model for binary outcome
    logit_model = sm.Logit(y, X)
    results = logit_model.fit(disp=False)

    # Calculate and print odds ratios with 95% confidence intervals for interpretability
    try:
        params = results.params
        conf = results.conf_int()
        or_df = pd.DataFrame({
            'OR': np.exp(params),
            'CI_lower': np.exp(conf[0]),
            'CI_upper': np.exp(conf[1])
        })
        print('Logit model summary:')
        print(results.summary())
        print('\nOdds ratios with 95% CI:')
        print(or_df)
    except Exception:
        # If printing fails for some reason, continue and return results
        pass

    # Return the fitted results object (contains coefficients, p-values, etc.)
    return results


