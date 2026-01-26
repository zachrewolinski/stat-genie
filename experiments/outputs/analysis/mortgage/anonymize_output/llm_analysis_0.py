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
    Transform raw dataset into analysis-ready dataframe.

    - Create clear, typed columns used by the model.
    - Handle missing values: drop rows missing DV or IV; impute controls (median for continuous, mode for binaries).
    - Standardize continuous controls (z-score) used in the model to aid numerical stability/interpretable coefficients.

    Returned dataframe contains at least the following columns (names used in modeling):
    ['Female','Approved','Black','LoanAmount','LoanAmount_z','HousingExpenseRatio','HousingExpenseRatio_z',
     'DebtToIncome','DebtToIncome_z','LTV','LTV_z','MortgageCreditScore','MortgageCreditScore_z',
     'ConsumerCreditScore','ConsumerCreditScore_z','BadCredit','SelfEmployed','Married','PMI_Denied']

    """
    df = df.copy()

    # Map / create clear column names
    df['Female'] = df['feature2']
    df['Black'] = df['feature3']
    df['HousingExpenseRatio'] = df['feature4']
    df['SelfEmployed'] = df['feature5']
    df['Married'] = df['feature6']
    df['MortgageCreditScore'] = df['feature7']
    df['ConsumerCreditScore'] = df['feature8']
    df['BadCredit'] = df['feature9']
    df['DebtToIncome'] = df['feature10']
    # feature11 is denial indicator; feature14 is acceptance indicator
    df['Approved'] = df['feature14']
    df['LTV'] = df['feature12']
    df['PMI_Denied'] = df['feature13']
    df['LoanAmount'] = df['feature1']

    # Ensure binary columns are integer 0/1
    binary_cols = ['Female','Black','SelfEmployed','Married','BadCredit','Approved','PMI_Denied']
    for c in binary_cols:
        # if column exists and is float, coerce to 0/1 where possible
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            # round floats close to 0/1
            df[c] = df[c].round().astype('Float64')

    # Drop rows missing DV or IV (we cannot impute target or primary predictor)
    df = df.dropna(subset=['Approved','Female'])

    # For continuous controls: impute median if missing
    continuous = ['LoanAmount','HousingExpenseRatio','DebtToIncome','LTV','MortgageCreditScore','ConsumerCreditScore']
    for c in continuous:
        if c in df.columns:
            med = df[c].median()
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = df[c].fillna(med)

    # For binary controls: impute mode (most common value) if missing
    binary_controls = ['Black','SelfEmployed','Married','BadCredit','PMI_Denied']
    for c in binary_controls:
        if c in df.columns:
            mode = df[c].mode()
            fill = int(mode.iloc[0]) if (mode is not None and len(mode) > 0) else 0
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(fill).astype(int)

    # Ensure Approved and Female are integers 0/1
    df['Approved'] = df['Approved'].astype(int)
    df['Female'] = df['Female'].astype(int)

    # Standardize continuous variables (z-score). Use sample std (ddof=0) to avoid division by zero in constant columns.
    for c in continuous:
        zname = c + '_z'
        col = df[c].astype(float)
        mean = col.mean()
        std = col.std(ddof=0)
        if std == 0 or np.isnan(std):
            # if no variance, set z to 0
            df[zname] = 0.0
        else:
            df[zname] = (col - mean) / std

    # Final check: drop rows with any remaining NA in model columns
    model_cols = ['Female','Approved','Black','SelfEmployed','Married','BadCredit','PMI_Denied',
                  'LoanAmount_z','HousingExpenseRatio_z','DebtToIncome_z','LTV_z',
                  'MortgageCreditScore_z','ConsumerCreditScore_z']
    # If any of these are missing (rare after imputation), drop the row
    df = df.dropna(subset=model_cols)

    # Keep only needed columns (but keep original features as well if user wants them)
    keep_cols = model_cols + ['LoanAmount','HousingExpenseRatio','DebtToIncome','LTV','MortgageCreditScore','ConsumerCreditScore']
    # Also keep Female and Approved explicitly
    keep_cols = list(dict.fromkeys(['Female','Approved'] + keep_cols))

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression to estimate effect of gender (Female) on mortgage approval (Approved),
    controlling for applicant financial and demographic covariates.

    Returns a dictionary with the fitted statsmodels Logit object under key 'model' and a DataFrame
    of odds ratios with 95% confidence intervals under key 'odds_ratios'.
    """
    # Columns to include in model (these must exist in transformed dataframe)
    cols = [
        'Female',
        'Black',
        'SelfEmployed',
        'Married',
        'BadCredit',
        'PMI_Denied',
        'LoanAmount_z',
        'HousingExpenseRatio_z',
        'DebtToIncome_z',
        'LTV_z',
        'MortgageCreditScore_z',
        'ConsumerCreditScore_z'
    ]

    # Ensure columns present
    missing = [c for c in cols + ['Approved'] if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Transformed dataframe is missing required columns for modeling: {missing}")

    X = df[cols].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['Approved'].astype(int)

    # Fit logistic regression (maximum likelihood)
    logit = sm.Logit(y, X).fit(disp=False)

    # Compute odds ratios and 95% confidence intervals
    params = logit.params
    conf = logit.conf_int()
    conf.columns = ['CI_lower', 'CI_upper']
    odds = np.exp(params)
    conf_odds = np.exp(conf)

    or_df = pd.DataFrame({
        'OR': odds,
        'CI_lower': conf_odds['CI_lower'],
        'CI_upper': conf_odds['CI_upper']
    })

    # Return model object and odds ratio table
    results = {
        'model': logit,
        'odds_ratios': or_df
    }
    return results


