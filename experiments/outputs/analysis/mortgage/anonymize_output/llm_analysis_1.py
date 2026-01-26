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
    Transform the raw Boston HMDA-like dataset into a cleaned dataframe with explicit column names
    used in the statistical model. Steps:
    - Rename features to meaningful column names used in modeling
    - Convert columns to numeric types
    - Drop rows with missing values in the DV (Accepted), IV (Female), or key controls
    - Ensure binary columns are integer (0/1)

    Returns the cleaned dataframe with these columns:
    ['Feature1','Female','Black','HousingExpenseRatio','SelfEmployed','Married',
     'MortgageCreditScore','ConsumerCreditScore','BadCredit','DebtToIncome','LTV','PMIDenied','Accepted']
    """

    df = df.copy()

    # Rename raw columns to model column names (exact names used downstream)
    rename_map = {
        'feature1': 'Feature1',
        'feature2': 'Female',
        'feature3': 'Black',
        'feature4': 'HousingExpenseRatio',
        'feature5': 'SelfEmployed',
        'feature6': 'Married',
        'feature7': 'MortgageCreditScore',
        'feature8': 'ConsumerCreditScore',
        'feature9': 'BadCredit',
        'feature10': 'DebtToIncome',
        'feature11': 'Denied',      # redundant with feature14; kept if present but not used
        'feature12': 'LTV',
        'feature13': 'PMIDenied',
        'feature14': 'Accepted'
    }
    df = df.rename(columns=rename_map)

    # Convert the columns of interest to numeric, coerce errors to NaN
    cols_to_numeric = ['Feature1','Female','Black','HousingExpenseRatio','SelfEmployed',
                       'Married','MortgageCreditScore','ConsumerCreditScore','BadCredit',
                       'DebtToIncome','LTV','PMIDenied','Accepted']
    for c in cols_to_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Require DV and IV to be present
    df = df.dropna(subset=['Accepted', 'Female'])

    # Require key controls present (drop rows with missing control values)
    required_controls = ['Black','MortgageCreditScore','ConsumerCreditScore','BadCredit',
                         'DebtToIncome','LTV','HousingExpenseRatio','SelfEmployed','Married','PMIDenied','Feature1']
    # Keep only controls that actually exist in the dataframe (robustness if some columns missing)
    existing_required = [c for c in required_controls if c in df.columns]
    if existing_required:
        df = df.dropna(subset=existing_required)

    # Ensure binary columns are integers (0/1)
    binary_cols = [c for c in ['Female','Black','SelfEmployed','Married','BadCredit','PMIDenied','Accepted'] if c in df.columns]
    for c in binary_cols:
        # round any floats that should be binary and cast
        df[c] = df[c].round().astype(int)
        # enforce 0/1 by clipping
        df[c] = df[c].clip(lower=0, upper=1)

    # For safety, reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression (logit) predicting mortgage approval (Accepted) from Female (gender)
    controlling for observed creditworthiness and applicant/property covariates.

    Returns a dictionary containing:
    - 'result': the fitted statsmodels LogitResults object
    - 'odds_ratios': a pandas DataFrame with ORs, 95% CI, and p-values for model coefficients
    """

    df = df.copy()

    # Ensure required columns exist
    required = ['Accepted', 'Female']
    if not all(col in df.columns for col in required):
        raise ValueError('Dataframe must contain at least Accepted and Female columns. Run transform() first.')

    # Build model matrix: include IV + all controls that exist in the dataframe
    control_cols = [
        'Black', 'HousingExpenseRatio', 'SelfEmployed', 'Married',
        'MortgageCreditScore', 'ConsumerCreditScore', 'BadCredit',
        'DebtToIncome', 'LTV', 'PMIDenied', 'Feature1'
    ]
    # Keep only columns present
    control_cols = [c for c in control_cols if c in df.columns]

    X_cols = ['Female'] + control_cols
    X = df[X_cols]
    y = df['Accepted']

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (use robust maxiter default)
    logit_model = sm.Logit(y, X)
    try:
        res = logit_model.fit(disp=False)
    except Exception as e:
        # Try a slightly different solver if convergence issues
        res = logit_model.fit(method='lbfgs', disp=False)

    # Compute odds ratios and 95% CIs
    params = res.params
    conf = res.conf_int()
    conf.columns = ['2.5%', '97.5%']
    odds_ratios = pd.DataFrame({
        'OR': np.exp(params),
        'CI_lower': np.exp(conf['2.5%']),
        'CI_upper': np.exp(conf['97.5%']),
        'pvalue': res.pvalues
    })

    # Order rows so intercept is last for readability
    if 'const' in odds_ratios.index:
        odds_ratios = odds_ratios.drop(index=['const']).append(odds_ratios.loc[['const']])

    return {
        'result': res,
        'odds_ratios': odds_ratios
    }


