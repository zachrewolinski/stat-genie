from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/shuffle_names_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataframe to produce the columns needed for the analysis.

    The function will:
    - Ensure required input columns exist.
    - Convert relevant columns to numeric (coercing errors).
    - Compute StudentTeacherRatio, AvgTestScore, LogEnrollment and rename/control variables.
    - Return dataframe with the exact column names used in the model.

    Notes:
    - AvgTestScore is computed as the mean of available standardized test scores
      (columns 'grades' and 'rownames'); rows with neither score are dropped.
    - We only drop rows that make it impossible to compute the required final
      modeling columns (e.g., non-positive teachers or enrollment, or missing
      expenditures or controls).
    """

    df = df.copy()

    # Required input columns presence (at least one test score column must exist)
    base_required = ['calworks', 'teachers', 'read', 'math', 'district']
    score_cols = ['grades', 'rownames']

    missing_base = [c for c in base_required if c not in df.columns]
    if missing_base:
        raise KeyError(f"The following required columns are missing from the input dataframe: {missing_base}")

    if not any(c in df.columns for c in score_cols):
        raise KeyError(f"At least one test score column ('grades' or 'rownames') must be present in the input dataframe.")

    # Coerce numeric for all potentially needed columns
    potential_numeric = list(set(base_required + [c for c in score_cols if c in df.columns]))
    for c in potential_numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Remove rows with non-positive teachers or non-positive enrollment (can't compute ratio)
    df = df[(df['teachers'] > 0) & (df['calworks'] > 0)]

    # Create canonical columns
    df['TotalEnrollment'] = df['calworks']
    df['Teachers'] = df['teachers']

    # Compute StudentTeacherRatio
    df['StudentTeacherRatio'] = df['TotalEnrollment'] / df['Teachers']

    # Compute AvgTestScore as mean of available test scores (skip missing)
    available_scores = [c for c in score_cols if c in df.columns]
    df['AvgTestScore'] = df[available_scores].mean(axis=1, skipna=True)

    # Expenditure per student and controls
    df['ExpenditurePerStudent'] = df['read']
    df['PctFreeLunch'] = df['math']
    df['PctEL'] = df['district']

    # Log enrollment
    df['LogEnrollment'] = np.log(df['TotalEnrollment'] + 1)

    # Now drop rows that are missing any of the final model columns
    keep_cols = [
        'StudentTeacherRatio',
        'AvgTestScore',
        'ExpenditurePerStudent',
        'PctFreeLunch',
        'PctEL',
        'LogEnrollment',
        'TotalEnrollment',
        'Teachers'
    ]

    # Ensure all keep_cols exist (they should) and are numeric where applicable
    for c in ['StudentTeacherRatio', 'AvgTestScore', 'ExpenditurePerStudent', 'PctFreeLunch', 'PctEL', 'LogEnrollment']:
        # Coerce to numeric to ensure consistent types
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Replace infinite values and drop rows with missing values in the model inputs
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[
        'StudentTeacherRatio',
        'AvgTestScore',
        'ExpenditurePerStudent',
        'PctFreeLunch',
        'PctEL',
        'LogEnrollment'
    ])

    return df[keep_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression of AvgTestScore on StudentTeacherRatio with controls.

    Model specification:
      AvgTestScore_i = beta0 + beta1 * StudentTeacherRatio_i
                       + beta2 * ExpenditurePerStudent_i
                       + beta3 * PctFreeLunch_i
                       + beta4 * PctEL_i
                       + beta5 * LogEnrollment_i + epsilon_i

    Returns:
      results: statsmodels regression results object with robust HC3 standard errors.
    """

    df = df.copy()

    model_cols = [
        'AvgTestScore',
        'StudentTeacherRatio',
        'ExpenditurePerStudent',
        'PctFreeLunch',
        'PctEL',
        'LogEnrollment'
    ]

    missing_model_cols = [c for c in model_cols if c not in df.columns]
    if missing_model_cols:
        raise KeyError(f"The following required model columns are missing from the dataframe: {missing_model_cols}")

    # Ensure numeric and drop non-finite
    for c in model_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=model_cols)

    if df.shape[0] == 0:
        raise ValueError("The transformed dataframe contains no observations after cleaning; cannot fit the model.")

    if df.shape[0] < 2:
        raise ValueError(f"Not enough observations to estimate the model (n={df.shape[0]}). Need at least 2 non-missing rows.")

    y = df['AvgTestScore'].astype(float)
    X = df[['StudentTeacherRatio', 'ExpenditurePerStudent', 'PctFreeLunch', 'PctEL', 'LogEnrollment']].astype(float)

    X = sm.add_constant(X, has_constant='add')

    # Fit OLS, then obtain robust HC3 covariance results
    ols_res = sm.OLS(y, X).fit()
    ols_robust = ols_res.get_robustcov_results(cov_type='HC3')

    return ols_robust