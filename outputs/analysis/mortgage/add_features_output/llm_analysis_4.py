from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/mortgage/add_features_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy to avoid modifying the original
    df = df.copy()

    # Columns required for analysis
    required_cols = [
        'accept', 'female', 'black', 'mortgage_credit', 'consumer_credit',
        'PI_ratio', 'loan_to_value', 'bad_history', 'married', 'self_employed',
        'housing_expense_ratio', 'denied_PMI'
    ]

    # Ensure required columns exist; if any are missing raise a clear error
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for transform: {missing}")

    # Drop rows with missing values in any of the model variables
    df = df.dropna(subset=required_cols)

    # Ensure binary columns are integer (0/1)
    for bcol in ['accept', 'female', 'black', 'bad_history', 'married', 'self_employed', 'denied_PMI']:
        # Cast to integers; if values aren't 0/1 this will convert truthy values appropriately
        df[bcol] = df[bcol].astype(int)

    # Create interaction for moderation test: female * black
    df['female_black'] = df['female'] * df['black']

    # Standardize continuous / ordinal numeric controls (z-scores) for easier comparison
    # Use sample std (ddof=1) via pandas .std()
    def zscore(series: pd.Series, name: str) -> pd.Series:
        s = series.astype(float)
        std = s.std()
        if std == 0 or np.isnan(std):
            # If constant or undefined std, return zeros to avoid division by zero
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    df['z_mortgage_credit'] = zscore(df['mortgage_credit'], 'mortgage_credit')
    df['z_consumer_credit'] = zscore(df['consumer_credit'], 'consumer_credit')
    df['z_PI_ratio'] = zscore(df['PI_ratio'], 'PI_ratio')
    df['z_loan_to_value'] = zscore(df['loan_to_value'], 'loan_to_value')
    df['z_housing_expense_ratio'] = zscore(df['housing_expense_ratio'], 'housing_expense_ratio')

    # Final dataframe contains all original columns plus derived columns used in the model
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # Ensure the transformed dataframe has the required columns
    required_model_cols = [
        'accept', 'female', 'black', 'female_black',
        'z_mortgage_credit','z_consumer_credit','z_PI_ratio','z_loan_to_value','z_housing_expense_ratio',
        'bad_history','married','self_employed','denied_PMI'
    ]
    missing = [c for c in required_model_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    # Define endogenous and exogenous variables
    endog = df['accept']

    exog_cols = [
        'female',            # IV
        'black',             # race control (and moderator)
        'female_black',      # interaction to test moderation
        'z_mortgage_credit',
        'z_consumer_credit',
        'z_PI_ratio',
        'z_loan_to_value',
        'z_housing_expense_ratio',
        'bad_history',
        'married',
        'self_employed',
        'denied_PMI'
    ]

    exog = df[exog_cols].astype(float)
    exog = sm.add_constant(exog, has_constant='add')

    # Fit a logistic regression (binomial GLM). Use robust (HC1) standard errors.
    model = sm.GLM(endog, exog, family=sm.families.Binomial())
    results = model.fit(cov_type='HC1')

    # Print a concise summary for quick inspection; return the full results object
    try:
        print(results.summary())
    except Exception:
        # summary() sometimes fails for certain result types; ignore printing failure
        pass

    return results


