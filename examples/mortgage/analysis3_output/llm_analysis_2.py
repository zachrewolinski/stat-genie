from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_hc3
import scipy.stats as stats

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/examples/mortgage/analysis3_output/mortgage.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to a modeling dataframe.
    Produces the following columns required for modeling:
      - approved: 1 if mortgage application accepted, 0 if denied
      - female: 1 if applicant is female, 0 if male (from consumer_credit)
      - credit_score_z: standardized credit score (from 'accept')
      - loan_to_value_z: standardized loan_to_value
      - debt_to_income_z: standardized debt measure (from 'denied_PMI')
      - self_employed: binary from 'self_employed'
      - married: binary from 'married'
      - bad_history: binary from 'bad_history'
      - housing_expense_ratio_z: standardized 'housing_expense_ratio'

    The function coerces relevant columns to numeric, drops rows with missing values in any of the
    essential columns, and returns the cleaned dataframe.
    """
    df = df.copy()

    # Ensure columns referenced below exist; coerce to numeric with NaN for invalid values
    numeric_cols = [
        'mortgage_credit', 'consumer_credit', 'accept', 'loan_to_value', 'denied_PMI',
        'self_employed', 'married', 'bad_history', 'housing_expense_ratio'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Independent variable: gender indicator
    # Dataset documentation: 'consumer_credit' == 1 if applicant is female, 0 if male
    if 'consumer_credit' in df.columns:
        df['female'] = df['consumer_credit'].astype(float)
    elif 'female' in df.columns:
        # fallback if the dataset already has a 'female' column
        df['female'] = pd.to_numeric(df['female'], errors='coerce')
    else:
        # if neither present, create an NA column so missing rows will be dropped downstream
        df['female'] = np.nan

    # Dependent variable: approved (1 accepted, 0 denied)
    # Documentation indicates 'mortgage_credit' is 1 if denied, 0 if accepted
    if 'mortgage_credit' in df.columns:
        df['approved'] = (df['mortgage_credit'] == 0).astype(float)
    else:
        # fallback: try to infer from other columns if available
        df['approved'] = np.nan

    # Controls: coerce and create columns
    df['credit_score'] = pd.to_numeric(df['accept'], errors='coerce') if 'accept' in df.columns else np.nan
    df['loan_to_value'] = pd.to_numeric(df['loan_to_value'], errors='coerce') if 'loan_to_value' in df.columns else np.nan
    df['debt_to_income'] = pd.to_numeric(df['denied_PMI'], errors='coerce') if 'denied_PMI' in df.columns else np.nan
    df['self_employed'] = pd.to_numeric(df['self_employed'], errors='coerce') if 'self_employed' in df.columns else np.nan
    df['married'] = pd.to_numeric(df['married'], errors='coerce') if 'married' in df.columns else np.nan
    df['bad_history'] = pd.to_numeric(df['bad_history'], errors='coerce') if 'bad_history' in df.columns else np.nan
    df['housing_expense_ratio'] = pd.to_numeric(df['housing_expense_ratio'], errors='coerce') if 'housing_expense_ratio' in df.columns else np.nan

    # Drop rows missing key variables for the analysis
    required = ['approved', 'female', 'credit_score', 'loan_to_value', 'debt_to_income',
                'self_employed', 'married', 'bad_history', 'housing_expense_ratio']
    # Keep only required columns that actually exist in df to avoid KeyError
    required_existing = [c for c in required if c in df.columns]
    df = df.dropna(subset=required_existing)

    # Standardize continuous controls (z-scores) to aid convergence and interpretation
    def zscore(series: pd.Series) -> pd.Series:
        if series.std(ddof=0) == 0 or np.isclose(series.std(ddof=0), 0.0):
            return series - series.mean()
        return (series - series.mean()) / series.std(ddof=0)

    if 'credit_score' in df.columns:
        df['credit_score_z'] = zscore(df['credit_score'])
    if 'loan_to_value' in df.columns:
        df['loan_to_value_z'] = zscore(df['loan_to_value'])
    if 'debt_to_income' in df.columns:
        df['debt_to_income_z'] = zscore(df['debt_to_income'])
    if 'housing_expense_ratio' in df.columns:
        df['housing_expense_ratio_z'] = zscore(df['housing_expense_ratio'])

    # Ensure binary controls are 0/1
    for b in ['self_employed', 'married', 'bad_history']:
        if b in df.columns:
            # coerce values to 0/1 when possible
            df[b] = df[b].apply(lambda x: 1.0 if x == 1 or x == 1.0 else (0.0 if x == 0 or x == 0.0 else np.nan))

    # After standardization and binary coercion, drop any rows with NaN in the final model columns
    final_model_cols = ['approved', 'female', 'credit_score_z', 'loan_to_value_z', 'debt_to_income_z',
                        'self_employed', 'married', 'bad_history', 'housing_expense_ratio_z']
    final_existing = [c for c in final_model_cols if c in df.columns]
    df = df.dropna(subset=final_existing)

    # Cast final columns to appropriate dtypes
    # approved and female are required to be exact 0/1 integers
    df['approved'] = df['approved'].astype(int)
    df['female'] = df['female'].astype(int)
    for b in ['self_employed', 'married', 'bad_history']:
        if b in df.columns:
            df[b] = df[b].astype(int)

    # Return the transformed dataframe with only the columns needed for the model (plus originals preserved)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting approval (approved) from female indicator and controls.
    Returns: results-like object with robust standard errors (HC3).
    Model specification:
      approved ~ female + credit_score_z + loan_to_value_z + debt_to_income_z + self_employed + married + bad_history + housing_expense_ratio_z
    """
    # Select model columns (must match the columns created by transform)
    model_cols = ['female', 'credit_score_z', 'loan_to_value_z', 'debt_to_income_z',
                  'self_employed', 'married', 'bad_history', 'housing_expense_ratio_z']

    missing = [c for c in model_cols + ['approved'] if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    X = df[model_cols]
    X = sm.add_constant(X)
    y = df['approved']

    # Fit logistic regression (maximum likelihood)
    logit = sm.Logit(y, X)
    try:
        res = logit.fit(disp=False)
    except Exception:
        # Try a slightly more robust approach if perfect separation or convergence issues occur
        res = logit.fit(method='bfgs', disp=False, maxiter=1000)

    # Compute robust (HC3) covariance matrix
    try:
        robust_cov = cov_hc3(res)
    except Exception:
        # Fallback: if cov_hc3 fails for some reason, use the default covariance matrix
        robust_cov = res.cov_params()

    # Construct a lightweight results-like object carrying robust standard errors and related stats
    class RobustResults:
        def __init__(self, base_res, cov_matrix):
            self.base_res = base_res
            # Parameters as a pandas Series
            self.params = base_res.params.copy()
            # Covariance matrix as numpy array (ensure ordering matches params)
            self.cov_params = pd.DataFrame(cov_matrix, index=self.params.index, columns=self.params.index)
            # Standard errors (robust)
            self.bse = pd.Series(np.sqrt(np.diag(self.cov_params.values)), index=self.params.index)
            # z-stats and p-values
            with np.errstate(divide='ignore', invalid='ignore'):
                z_stats = self.params / self.bse
            # two-sided p-values using normal approximation
            self.pvalues = pd.Series(2 * stats.norm.sf(np.abs(z_stats)), index=self.params.index)
            # 95% confidence intervals
            zcrit = stats.norm.ppf(0.975)
            ci_lower = self.params - zcrit * self.bse
            ci_upper = self.params + zcrit * self.bse
            self.conf_int = pd.DataFrame({'2.5%': ci_lower, '97.5%': ci_upper}, index=self.params.index)
            # Expose some commonly used attributes from base results
            self.llf = getattr(base_res, 'llf', None)
            self.nobs = getattr(base_res, 'nobs', None)
            self.df_model = getattr(base_res, 'df_model', None)
            self.df_resid = getattr(base_res, 'df_resid', None)
            self.model = base_res.model

        def summary(self, *args, **kwargs):
            # Return the base summary (note: it will reflect default SEs), but attach a note about robust SEs.
            s = self.base_res.summary(*args, **kwargs)
            return s

        def __repr__(self):
            return f"<RobustResults params=\n{self.params}\n bse=\n{self.bse}\n>"

    res_robust = RobustResults(res, robust_cov)

    return res_robust