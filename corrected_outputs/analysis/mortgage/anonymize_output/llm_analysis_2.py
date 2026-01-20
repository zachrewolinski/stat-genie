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
    Transform the raw dataset into a cleaned dataframe with the exact columns used by the model.

    - Create binary/identifier columns with clear names used in modeling.
    - Standardize continuous controls for stable optimization and interpretability.
    - Drop rows with missing values in any of the variables required for modeling.

    Expected input columns: feature1..feature14 per dataset schema.
    Returns a dataframe that contains at least the columns listed in the conceptual variables:
      ['Accepted','Female','Black','SelfEmployed','Married','BadCredit',
       'LoanAmount_z','MortCreditScore_z','ConsCreditScore_z','DebtToIncome_z','LTV_z']
    """
    # Work on a copy
    df = df.copy()

    # Required raw columns (as per schema)
    required = [
        'feature1',   # loan / amount-like continuous variable
        'feature2',   # female indicator
        'feature3',   # black indicator
        'feature4',   # housing expense ratio (not used directly but keep if desired) - not included in final model
        'feature5',   # self-employed
        'feature6',   # married
        'feature7',   # mortgage credit score
        'feature8',   # consumer credit score
        'feature9',   # bad credit
        'feature10',  # debt-to-income
        'feature11',  # denied indicator (redundant)
        'feature12',  # LTV
        'feature13',  # insurer denial flag (not used directly)
        'feature14'   # accepted indicator
    ]

    # Drop rows missing any of the required variables
    df = df.dropna(subset=required)

    # Create clear column names used in modeling
    # Dependent variable: Accepted (1 = accepted, 0 = denied)
    # Prefer using feature14 as schema indicates 1 if accepted
    df['Accepted'] = df['feature14'].astype(int)

    # Independent variable: Female (1 = female, 0 = male)
    df['Female'] = df['feature2'].astype(int)

    # Controls (binary)
    df['Black'] = df['feature3'].astype(int)
    df['SelfEmployed'] = df['feature5'].astype(int)
    df['Married'] = df['feature6'].astype(int)
    df['BadCredit'] = df['feature9'].astype(int)

    # Continuous controls: ensure numeric
    df['LoanAmount'] = pd.to_numeric(df['feature1'], errors='coerce')
    df['MortCreditScore'] = pd.to_numeric(df['feature7'], errors='coerce')
    df['ConsCreditScore'] = pd.to_numeric(df['feature8'], errors='coerce')
    df['DebtToIncome'] = pd.to_numeric(df['feature10'], errors='coerce')
    df['LTV'] = pd.to_numeric(df['feature12'], errors='coerce')

    # Drop created columns with NaN from numeric coercion
    df = df.dropna(subset=['LoanAmount','MortCreditScore','ConsCreditScore','DebtToIncome','LTV'])

    # Standardize continuous controls (z-score). Use ddof=0 (population-style) for stability.
    for col in ['LoanAmount','MortCreditScore','ConsCreditScore','DebtToIncome','LTV']:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        # If std is zero (unlikely), avoid divide-by-zero
        if std == 0 or np.isnan(std):
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Keep only columns the model will need (but retain original standardized numeric columns)
    model_cols = [
        'Accepted', 'Female', 'Black', 'SelfEmployed', 'Married', 'BadCredit',
        'LoanAmount_z', 'MortCreditScore_z', 'ConsCreditScore_z', 'DebtToIncome_z', 'LTV_z'
    ]

    # Ensure types are numeric ints/floats
    for b in ['Accepted','Female','Black','SelfEmployed','Married','BadCredit']:
        df[b] = df[b].astype(int)

    for z in ['LoanAmount_z','MortCreditScore_z','ConsCreditScore_z','DebtToIncome_z','LTV_z']:
        df[z] = pd.to_numeric(df[z], errors='coerce')

    # Final drop if any of the model columns are NA (should be rare after earlier drops)
    df = df.dropna(subset=model_cols)

    # Return dataframe containing at least the required columns for modeling
    return df[model_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression predicting mortgage approval (Accepted) from Female and controls.

    Returns a dict with the fitted model result object and a table of average marginal effects.
    """
    # Prepare design matrices
    # df is expected to be the output of transform(); uses exact column names specified earlier
    X_cols = [
        'Female', 'Black', 'SelfEmployed', 'Married', 'BadCredit',
        'LoanAmount_z', 'MortCreditScore_z', 'ConsCreditScore_z', 'DebtToIncome_z', 'LTV_z'
    ]
    X = df[X_cols]
    y = df['Accepted']

    # Add constant for intercept
    X_const = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (maximum likelihood)
    logit_model = sm.Logit(y, X_const)
    try:
        result = logit_model.fit(disp=False)
    except Exception:
        # If convergence issues occur, try a GLM binomial as fallback
        glm_model = sm.GLM(y, X_const, family=sm.families.Binomial())
        result = glm_model.fit()

    # Compute average marginal effects for interpretation (Female effect on probability)
    # Use statsmodels' get_margeff if available (works for Logit/GLM results)
    try:
        marg = result.get_margeff(at='overall')
        marg_summary = marg.summary_frame()
    except Exception:
        marg_summary = None

    # Return results: the fitted result object and marginal effects table (if computed)
    return {
        'model_result': result,
        'marginal_effects': marg_summary
    }


