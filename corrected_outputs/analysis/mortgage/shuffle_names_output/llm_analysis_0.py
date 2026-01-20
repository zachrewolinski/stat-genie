from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/mortgage/shuffle_names_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe suitable for modeling.

    Outputs (required columns added/created):
    - approved: binary outcome (1 = approved/accepted, 0 = denied)
    - is_female: binary indicator (1 = female, 0 = male)
    - accept, loan_to_value, housing_expense_ratio, denied_PMI, PI_ratio, self_employed, bad_history:
      control columns forced to numeric; if not present in input they are created with NaN so the final dataframe has consistent columns.

    The function is robust to small differences in column naming as described in the dataset schema.
    """
    df = df.copy()

    # 1) Construct the dependent variable 'approved'. Prefer 'Unnamed: 0' (1 if accepted), otherwise invert 'mortgage_credit' (1 if denied).
    if 'Unnamed: 0' in df.columns:
        df['approved'] = pd.to_numeric(df['Unnamed: 0'], errors='coerce')
    elif 'mortgage_credit' in df.columns:
        # mortgage_credit is documented as 1 if denied, 0 if accepted -> approved = 1 - mortgage_credit
        df['approved'] = 1 - pd.to_numeric(df['mortgage_credit'], errors='coerce')
    else:
        # If neither column exists, create the column with NaN so the pipeline fails clearly in modeling step
        df['approved'] = np.nan

    # 2) Construct the independent variable 'is_female'. Prefer 'consumer_credit' which (per schema) encodes gender as 1 = female, 0 = male.
    if 'consumer_credit' in df.columns:
        df['is_female'] = pd.to_numeric(df['consumer_credit'], errors='coerce')
    elif 'female' in df.columns:
        # If a non-binary 'female' column exists (some datasets encode proportion or probability), convert to a binary indicator using 0.5 threshold
        df['is_female'] = (pd.to_numeric(df['female'], errors='coerce') > 0.5).astype(float)
    else:
        df['is_female'] = np.nan

    # 3) Ensure control columns exist in the final dataframe and are numeric. If a control is missing in the raw data, create it with NaN.
    control_cols = ['accept', 'loan_to_value', 'housing_expense_ratio', 'denied_PMI', 'PI_ratio', 'self_employed', 'bad_history']
    for c in control_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan

    # 4) Basic cleaning: keep rows where outcome and IV are observed. We'll leave rows with missing controls present; the model function will drop rows with missing predictors.
    df = df.dropna(subset=['approved', 'is_female']).reset_index(drop=True)

    # 5) Cast approved and is_female to integers where possible (they may remain float if NaN existed)
    df['approved'] = df['approved'].astype(float)
    df['is_female'] = df['is_female'].astype(float)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (binomial GLM) of mortgage approval on applicant gender and controls.

    Model specification:
    approved ~ is_female + accept + loan_to_value + housing_expense_ratio + denied_PMI + PI_ratio + self_employed + bad_history

    Returns robust (HC3) covariance results.
    """
    # Make a copy to avoid side effects
    df = df.copy()

    # Define modeling columns (predictors) and outcome
    predictors = ['is_female', 'accept', 'loan_to_value', 'housing_expense_ratio', 'denied_PMI', 'PI_ratio', 'self_employed', 'bad_history']
    outcome = 'approved'

    # Drop rows with missing outcome or missing predictors (list of predictors above)
    required = [outcome] + predictors
    df_model = df.dropna(subset=required).reset_index(drop=True)

    if df_model.shape[0] == 0:
        raise ValueError('No rows available for modeling after dropping missing values. Check transform() output and required columns.')

    # Prepare design matrix
    X = df_model[predictors]
    X = sm.add_constant(X, has_constant='add')
    y = df_model[outcome]

    # Fit binomial GLM (logistic regression)
    glm = sm.GLM(y, X, family=sm.families.Binomial())
    res = glm.fit()

    # Compute robust (HC3) standard errors and return the robust results wrapper
    try:
        res_robust = res.get_robustcov_results(cov_type='HC3')
    except Exception:
        # If robust results can't be computed for some reason, return the original fit
        res_robust = res

    # Print summary for convenience and return the results object
    print(res_robust.summary())
    return res_robust


