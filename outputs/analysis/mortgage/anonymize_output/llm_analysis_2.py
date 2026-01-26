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
    Clean and prepare the Boston mortgage dataset for modeling.
    Outputs a dataframe containing the exact column names used in the model:
      - approved (DV)
      - female (IV)
      - black, housing_expense_ratio_z, self_employed, married,
        mortgage_credit_score_z, consumer_credit_score_z, bad_credit,
        debt_to_income_z, loan_to_value_z, pmi_denied (controls)
    Steps:
      - construct approved flag (use feature14 if present; otherwise derive from feature11)
      - coerce binary predictors to 0/1 ints (nullable Int64 to allow NA)
      - standardize continuous predictors to z-scores (robust to zero std)
      - drop rows with missing values on required variables
    """
    df = df.copy()

    # --- Dependent variable: approved ---
    # feature14 is coded 1 = accepted, 0 = denied (per schema). If missing, use 1 - feature11 (feature11: 1 denied)
    if 'feature14' in df.columns:
        df['approved'] = df['feature14']
    elif 'feature11' in df.columns:
        df['approved'] = 1 - df['feature11']
    else:
        raise KeyError('Neither feature14 (accepted) nor feature11 (denied) present in dataframe')

    # Ensure approved is numeric 0/1
    df['approved'] = pd.to_numeric(df['approved'], errors='coerce').astype('float')

    # --- Independent variable: female ---
    if 'feature2' not in df.columns:
        raise KeyError('feature2 (gender) not present in dataframe')
    df['female'] = pd.to_numeric(df['feature2'], errors='coerce').astype('float')

    # --- Controls: binary flags ---
    # race: Black indicator
    df['black'] = pd.to_numeric(df['feature3'], errors='coerce').astype('float') if 'feature3' in df.columns else np.nan

    # housing expense ratio (continuous)
    df['housing_expense_ratio'] = pd.to_numeric(df['feature4'], errors='coerce') if 'feature4' in df.columns else np.nan

    # self-employed
    df['self_employed'] = pd.to_numeric(df['feature5'], errors='coerce').astype('float') if 'feature5' in df.columns else np.nan

    # married
    df['married'] = pd.to_numeric(df['feature6'], errors='coerce').astype('float') if 'feature6' in df.columns else np.nan

    # mortgage credit score (feature7) and consumer credit score (feature8)
    df['mortgage_credit_score'] = pd.to_numeric(df['feature7'], errors='coerce') if 'feature7' in df.columns else np.nan
    df['consumer_credit_score'] = pd.to_numeric(df['feature8'], errors='coerce') if 'feature8' in df.columns else np.nan

    # bad credit history
    df['bad_credit'] = pd.to_numeric(df['feature9'], errors='coerce').astype('float') if 'feature9' in df.columns else np.nan

    # debt to income
    df['debt_to_income'] = pd.to_numeric(df['feature10'], errors='coerce') if 'feature10' in df.columns else np.nan

    # loan to value
    df['loan_to_value'] = pd.to_numeric(df['feature12'], errors='coerce') if 'feature12' in df.columns else np.nan

    # private mortgage insurance denied flag
    df['pmi_denied'] = pd.to_numeric(df['feature13'], errors='coerce').astype('float') if 'feature13' in df.columns else np.nan

    # --- Drop rows missing the DV or the IV or key numeric controls ---
    required = [
        'approved', 'female', 'mortgage_credit_score', 'consumer_credit_score',
        'debt_to_income', 'loan_to_value'
    ]
    df = df.dropna(subset=required).reset_index(drop=True)

    # Ensure binary columns are 0/1 integers (where applicable)
    # Use pandas nullable integer dtype 'Int64' to allow NA values where controls were not provided
    for col in ['female', 'black', 'self_employed', 'married', 'bad_credit', 'pmi_denied']:
        if col in df.columns:
            # round & clip to 0/1 (handles floats like 0.0/1.0), keep NA as <NA> using Int64
            df[col] = df[col].round().clip(0, 1).astype('Int64')

    # --- Standardize continuous controls to z-scores with protection for zero std ---
    def safe_zscore(series: pd.Series) -> pd.Series:
        s = series.astype('float')
        mean = s.mean()
        std = s.std(ddof=0)
        if pd.isna(std) or std == 0:
            return s - mean  # will be all zeros if std == 0
        return (s - mean) / std

    df['housing_expense_ratio_z'] = safe_zscore(df['housing_expense_ratio'])
    df['debt_to_income_z'] = safe_zscore(df['debt_to_income'])
    df['loan_to_value_z'] = safe_zscore(df['loan_to_value'])
    df['mortgage_credit_score_z'] = safe_zscore(df['mortgage_credit_score'])
    df['consumer_credit_score_z'] = safe_zscore(df['consumer_credit_score'])

    # Keep only the columns we will use in the model (clean and explicit)
    model_cols = [
        'approved', 'female', 'black', 'housing_expense_ratio_z', 'self_employed', 'married',
        'mortgage_credit_score_z', 'consumer_credit_score_z', 'bad_credit', 'debt_to_income_z',
        'loan_to_value_z', 'pmi_denied'
    ]

    # Some control columns may not exist in the original dataset; ensure they are present with NaN if missing
    for col in model_cols:
        if col not in df.columns:
            df[col] = np.nan

    df_out = df[model_cols].copy()

    # Final drop: remove rows with any remaining missing required DV/IV values
    df_out = df_out.dropna(subset=['approved', 'female']).reset_index(drop=True)
    # It's acceptable to allow some rows with missing optional controls; for modeling we will drop those in model().
    return df_out


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression predicting mortgage approval from gender controlling for applicant characteristics.

    Model specification (primary):
      logit(P(approved=1)) = beta0 + beta1*female + sum(beta_k * control_k)

    Returns a dict with:
      - model: the fitted statsmodels LogitResults object
      - summary: textual model summary
      - odds_ratios: pandas Series of exp(params)
      - pvalues: pandas Series of p-values

    Notes:
      - The transform() function should have been applied before calling this function.
      - This function will drop rows with missing values in predictors used here.
    """
    df = df.copy()

    # Columns used in the model (must match transform output names)
    predictor_cols = [
        'female', 'black', 'self_employed', 'married',
        'mortgage_credit_score_z', 'consumer_credit_score_z', 'bad_credit',
        'debt_to_income_z', 'loan_to_value_z', 'housing_expense_ratio_z', 'pmi_denied'
    ]

    # Drop rows with missing values in any predictor (complete-case analysis)
    needed = ['approved'] + predictor_cols
    df_model = df.dropna(subset=needed).copy()

    if df_model.shape[0] == 0:
        raise ValueError('No rows available after dropping missing values for modeling. Check transform() output and required columns.')

    y = df_model['approved'].astype('int')
    X = df_model[predictor_cols].astype('float')

    # Add constant for intercept
    X = sm.add_constant(X, has_constant='add')

    # Fit logistic regression (maximum likelihood)
    # Use try/except to return informative error if convergence problems occur
    try:
        logit_model = sm.Logit(y, X)
        res = logit_model.fit(disp=False, method='lbfgs')
    except Exception as e:
        # Try a more robust fit (Newton) if lbfgs fails
        try:
            res = logit_model.fit(disp=False, method='newton')
        except Exception as e2:
            raise RuntimeError(f'Logit failed to converge with errors: {e}; {e2}')

    # Prepare results for return
    odds_ratios = np.exp(res.params)

    results = {
        'model': res,
        'summary': res.summary().as_text(),
        'odds_ratios': odds_ratios,
        'pvalues': res.pvalues,
        'n_obs': int(res.nobs)
    }

    return results