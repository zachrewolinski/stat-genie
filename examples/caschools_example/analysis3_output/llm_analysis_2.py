from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/caschools/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district-level dataframe to the modeling dataframe.

    Produces the following new/derived columns used in modeling:
      - StudentTeacherRatio: calworks / teachers (enrollment per teacher)
      - AvgTestScore: mean of 'grades' and 'rownames' (reading & math average), allowing one missing score
      - *_z: z-score standardized versions of continuous predictors used in the model
      - school_KK_08: dummy for the 'school' category 'KK_08' (created if present); otherwise 0

    The final dataframe will contain the exact columns required by the analysis contract:
      - StudentTeacherRatio_z
      - AvgTestScore
      - expenditure_z
      - income_z
      - district_z
      - computer_z
      - school_KK_08
    """
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric where expected
    numeric_cols = ['calworks', 'teachers', 'grades', 'rownames', 'expenditure', 'income', 'district', 'computer']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # We require calworks and teachers to compute the StudentTeacherRatio.
    # For the outcome AvgTestScore, we allow having at least one of 'grades' or 'rownames'.
    score_cols = [c for c in ['grades', 'rownames'] if c in df.columns]

    # Drop rows missing the key variables needed to compute the IV (calworks and teachers)
    required_iv = [c for c in ['calworks', 'teachers'] if c in df.columns]
    if required_iv:
        df = df.dropna(subset=required_iv)

    # Also drop rows where both score columns are missing (we need at least one score)
    if score_cols:
        df = df.dropna(subset=score_cols, how='all')

    # Remove implausible teacher counts (<=0) to avoid division by zero
    if 'teachers' in df.columns:
        df = df[df['teachers'] > 0]

    # Compute student-teacher ratio (only for rows that survived the above)
    if 'calworks' in df.columns and 'teachers' in df.columns:
        df['StudentTeacherRatio'] = df['calworks'] / df['teachers']
    else:
        # If either is absent, create the column but fill with NaN so downstream logic can handle it
        df['StudentTeacherRatio'] = np.nan

    # Compute average test score as mean of reading and math, allowing one to be missing
    if score_cols:
        df['AvgTestScore'] = df[score_cols].mean(axis=1, skipna=True)
    else:
        df['AvgTestScore'] = np.nan

    # Standardize continuous predictors (z-scores). Use ddof=0 for population-like standardization.
    def zscore(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        mean = s.mean()
        std = s.std(ddof=0)
        if pd.isna(std) or std == 0:
            # If no variation (or all missing), return zeros for non-missing and 0 for missing as well.
            res = s.copy()
            # center to zero and set non-missing to 0
            res = res - mean
            res.loc[~res.isna()] = 0.0
            res = res.fillna(0.0)
            return res
        return (s - mean) / std

    # Create z-scores for IV
    df['StudentTeacherRatio_z'] = zscore(df['StudentTeacherRatio'])

    # Controls: expenditure, income, district, computer
    # If a control is present, compute z-score. If absent, create a column filled with 0.0 so the final dataframe
    # contains the required column but does not cause all rows to be dropped.
    for c in ['expenditure', 'income', 'district', 'computer']:
        zcol = c + '_z'
        if c in df.columns:
            df[zcol] = zscore(df[c])
        else:
            df[zcol] = 0.0

    # Clean up school category strings to produce safe dummy column names and ensure the specific dummy exists
    # The analysis requires the final column 'school_KK_08' to be present.
    if 'school' in df.columns:
        # Fill NA first, then convert to string and sanitize
        df['school'] = df['school'].fillna('NA').astype(str).str.replace('[^0-9A-Za-z_]+', '_', regex=True)
        school_dummies = pd.get_dummies(df['school'], prefix='school', drop_first=True)
        # Ensure the specific required dummy exists; if not, create it with zeros
        if 'school_KK_08' not in school_dummies.columns:
            school_dummies['school_KK_08'] = 0
        df = pd.concat([df, school_dummies], axis=1)
    else:
        # If school not present, create the required dummy column filled with zeros
        df['school'] = 'NA'
        df['school_KK_08'] = 0

    # Final: keep only rows with non-missing values in the core model columns (IV and DV).
    # We require StudentTeacherRatio_z and AvgTestScore to be non-missing; controls exist (possibly filled with 0).
    df = df.dropna(subset=['StudentTeacherRatio_z', 'AvgTestScore'])

    # Ensure the final dataframe contains exactly the required conceptual variable columns (names must match).
    required_final_cols = [
        'StudentTeacherRatio_z',
        'AvgTestScore',
        'expenditure_z',
        'income_z',
        'district_z',
        'computer_z',
        'school_KK_08'
    ]
    for col in required_final_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model predicting AvgTestScore from StudentTeacherRatio (z-scored) controlling for expenditure, income,
    percent English learners (district), computer resources, and the school grade-span dummy 'school_KK_08'.

    Returns a fitted statsmodels regression results object (with robust HC3 standard errors).
    """
    # Copy to avoid side-effects
    df = df.copy()

    # Define the exact predictors required by the conceptual variables contract
    predictor_cols = [
        'StudentTeacherRatio_z',
        'expenditure_z',
        'income_z',
        'district_z',
        'computer_z',
        'school_KK_08'
    ]

    # Ensure all predictors exist in df
    missing = [c for c in predictor_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing predictors in dataframe required for model: {missing}")

    # Outcome and design matrix
    y = df['AvgTestScore']
    X = df[predictor_cols]

    # Drop rows with any missing values among y or X
    combined = pd.concat([y, X], axis=1).dropna()
    if combined.shape[0] == 0:
        raise ValueError("No observations available after dropping missing values for modeling.")

    y = combined['AvgTestScore']
    X = combined[predictor_cols]

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Fit OLS
    results = sm.OLS(y, X).fit()

    # Return results adjusted for robust HC3 standard errors
    try:
        robust_results = results.get_robustcov_results(cov_type='HC3')
    except Exception:
        # If for some reason robust cov can't be computed, return the plain results
        robust_results = results

    return robust_results