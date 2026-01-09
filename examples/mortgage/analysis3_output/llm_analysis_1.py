from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/mortgage/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # 1) Create dependent variable 'approved' (1 accepted, 0 denied).
    # Prefer 'Unnamed: 0' if it encodes acceptance; otherwise fall back to mortgage_credit (1=denied -> convert).
    approved_series = None

    if 'Unnamed: 0' in df.columns:
        tmp = pd.to_numeric(df['Unnamed: 0'], errors='coerce')
        uniq = set(tmp.dropna().unique())
        if len(uniq) > 0 and uniq.issubset({0, 1}):
            # Assume Unnamed: 0 encodes acceptance with 1 = accepted
            approved_series = (tmp == 1).astype(float)
        else:
            approved_series = None

    if approved_series is None and 'mortgage_credit' in df.columns:
        mc = pd.to_numeric(df['mortgage_credit'], errors='coerce')
        uniq_mc = set(mc.dropna().unique())
        if len(uniq_mc) > 0 and uniq_mc.issubset({0, 1}):
            # mortgage_credit: 1 = denied per dataset note -> approved = (mc == 0)
            approved_series = (mc == 0).astype(float)
        else:
            # If mortgage_credit is numeric but not strictly 0/1, treat values <= 0.5 as accepted
            # (this is a robust fallback to binarize noisy encodings)
            approved_series = (mc <= 0.5).astype(float)

    if approved_series is None:
        raise KeyError("Cannot derive 'approved' from 'Unnamed: 0' or 'mortgage_credit'.")

    df['approved'] = approved_series

    # 2) Create independent variable 'is_female'. Prefer 'female' column; otherwise use 'consumer_credit' as a backup.
    if 'female' in df.columns:
        f = pd.to_numeric(df['female'], errors='coerce')
        df['is_female'] = (f > 0.5).astype(int)
    elif 'consumer_credit' in df.columns:
        cc = pd.to_numeric(df['consumer_credit'], errors='coerce')
        df['is_female'] = (cc > 0.5).astype(int)
    else:
        raise KeyError("No column available to derive gender ('female' or 'consumer_credit').")

    # 3) Ensure presence and numeric typing for control columns. If missing, create with NaNs so they can be dropped later.
    control_cols = ['bad_history', 'loan_to_value', 'PI_ratio', 'housing_expense_ratio',
                    'married', 'self_employed', 'denied_PMI', 'black']
    for col in control_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    # 4) Derive a consumer credit / score proxy from 'accept' if present (dataset uses 'accept' in some versions)
    if 'accept' in df.columns:
        df['consumer_score'] = pd.to_numeric(df['accept'], errors='coerce')
    else:
        # if no 'accept', create NA column
        df['consumer_score'] = np.nan

    # 5) Standardize continuous numeric controls to z-scores (to help model stability/interpretation)
    # Columns standardized: PI_ratio, housing_expense_ratio, denied_PMI, consumer_score
    cont_to_z = {
        'PI_ratio': 'z_PI_ratio',
        'housing_expense_ratio': 'z_housing_expense_ratio',
        'denied_PMI': 'z_denied_PMI',
        'consumer_score': 'z_consumer_score'
    }
    for orig, zcol in cont_to_z.items():
        series = pd.to_numeric(df[orig], errors='coerce')
        mean = series.mean(skipna=True)
        std = series.std(ddof=0, skipna=True)
        if std == 0 or np.isnan(std):
            # if constant or missing, keep NaNs
            df[zcol] = np.nan
        else:
            df[zcol] = (series - mean) / (std)

    # 6) Select final columns required for modeling
    final_cols = ['approved', 'is_female', 'bad_history', 'loan_to_value', 'married',
                  'self_employed', 'black', 'z_PI_ratio', 'z_housing_expense_ratio',
                  'z_denied_PMI', 'z_consumer_score']

    df_final = df[final_cols].copy()

    # 7) Drop rows with missing values in any of the modeling columns
    df_final = df_final.dropna()

    # Ensure types: approved and is_female as integers, controls numeric
    df_final['approved'] = pd.to_numeric(df_final['approved'], errors='coerce').astype(int)
    df_final['is_female'] = pd.to_numeric(df_final['is_female'], errors='coerce').astype(int)

    return df_final


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression predicting approval from gender controlling for creditworthiness and demographics.
    Returns a dictionary with the fitted statsmodels Logit result and a DataFrame of odds ratios with 95% CI.
    """
    df = df.copy()

    # Ensure input dataframe has the columns produced by transform
    required = ['approved', 'is_female', 'bad_history', 'loan_to_value', 'married',
                'self_employed', 'black', 'z_PI_ratio', 'z_housing_expense_ratio',
                'z_denied_PMI', 'z_consumer_score']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Transformed dataframe is missing required columns: {missing}")

    # Dependent and independent variables
    y = pd.to_numeric(df['approved'], errors='coerce').astype(float)
    if y.isna().any():
        raise ValueError("Dependent variable 'approved' contains NaN after coercion.")
    if y.min() < 0 or y.max() > 1:
        raise ValueError("Dependent variable 'approved' must be binary in {0,1} for logistic regression.")

    X = df[['is_female', 'bad_history', 'loan_to_value', 'married',
            'self_employed', 'black', 'z_PI_ratio', 'z_housing_expense_ratio',
            'z_denied_PMI', 'z_consumer_score']].astype(float)

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (maximum likelihood)
    logit_model = sm.Logit(y, X)
    result = logit_model.fit(disp=False)

    # Compute odds ratios and 95% CI
    params = result.params
    conf = result.conf_int()
    odds_ratios = np.exp(params)
    conf_odds = np.exp(conf)
    odds_df = pd.DataFrame({
        'odds_ratio': odds_ratios,
        'ci_lower': conf_odds.iloc[:, 0],
        'ci_upper': conf_odds.iloc[:, 1]
    })

    # Return fitted result and odds ratio table
    return {
        'model_result': result,
        'odds_ratios': odds_df
    }