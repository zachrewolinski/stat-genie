from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/shuffle_names_output/caschools.csv')


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataframe into FINAL dataframe containing the exact columns:
      - StudentTeacherRatio
      - AvgScore
      - ExpenditurePerStudent
      - PctFreeLunch
      - PctEnglishLearners
      - ComputersPerClassroom

    The function will:
      - Coerce likely numeric source columns to numeric.
      - Create the required final columns (even if filled with NaN when source missing).
      - Compute StudentTeacherRatio when possible (calworks / teachers).
      - Compute AvgScore as the mean of available standardized reading/math columns
        (uses 'grades' and 'rownames' when available).
      - Preserve rows (only drop rows that lack both the primary IV and DV), so that
        model can handle missing controls via statsmodels' missing='drop'.
    """
    df = df.copy()

    # Coerce candidate columns to numeric if present
    candidate_numeric = [
        'calworks', 'teachers', 'grades', 'rownames',
        'read', 'math', 'district', 'english'
    ]
    for c in candidate_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Initialize required final columns to ensure they exist in output
    df['StudentTeacherRatio'] = np.nan
    df['AvgScore'] = np.nan
    df['ExpenditurePerStudent'] = np.nan
    df['PctFreeLunch'] = np.nan
    df['PctEnglishLearners'] = np.nan
    df['ComputersPerClassroom'] = np.nan

    # Compute StudentTeacherRatio if possible (students per FTE teacher)
    if ('calworks' in df.columns) and ('teachers' in df.columns):
        # Avoid division by zero and require positive teachers
        valid_mask = df['teachers'].notna() & (df['teachers'] > 0) & df['calworks'].notna()
        df.loc[valid_mask, 'StudentTeacherRatio'] = df.loc[valid_mask, 'calworks'] / df.loc[valid_mask, 'teachers']

    # Compute AvgScore as mean of available reading/math district averages.
    # Prefer 'grades' and 'rownames' as indicated by dataset schema; use whichever are present.
    score_sources = [c for c in ['grades', 'rownames'] if c in df.columns]
    if score_sources:
        df['AvgScore'] = df[score_sources].mean(axis=1)

    # Map controls from source columns to required final column names (keep NaN if source absent)
    if 'read' in df.columns:
        df['ExpenditurePerStudent'] = df['read']
    if 'math' in df.columns:
        df['PctFreeLunch'] = df['math']
    if 'district' in df.columns:
        df['PctEnglishLearners'] = df['district']
    if 'english' in df.columns:
        df['ComputersPerClassroom'] = df['english']

    # Optional helper: log transform of StudentTeacherRatio (not used by model by default)
    # Only compute where StudentTeacherRatio is positive
    df['LogStudentTeacherRatio'] = np.nan
    positive_mask = df['StudentTeacherRatio'].notna() & (df['StudentTeacherRatio'] > 0)
    df.loc[positive_mask, 'LogStudentTeacherRatio'] = np.log(df.loc[positive_mask, 'StudentTeacherRatio'])

    # Final: ensure the FINAL dataframe contains the required columns (already created).
    # Drop rows that are missing the essential IV or DV (model cannot run without them).
    # Controls may be missing; statsmodels will drop rows with missing values when fitting.
    df = df.dropna(subset=['StudentTeacherRatio', 'AvgScore'])

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit OLS of AvgScore on StudentTeacherRatio and controls with robust SEs.
    Required columns (must be present in the FINAL dataframe):
      - StudentTeacherRatio (IV)
      - AvgScore (DV)
      - ExpenditurePerStudent, PctFreeLunch, PctEnglishLearners, ComputersPerClassroom (controls; may be all-NaN)
    The function validates inputs and raises a clear error if there are no usable observations.
    """
    df = df.copy()

    # Define predictors in the required order and keep those columns present in the dataframe
    predictors = [
        'StudentTeacherRatio',
        'ExpenditurePerStudent',
        'PctFreeLunch',
        'PctEnglishLearners',
        'ComputersPerClassroom'
    ]
    # Ensure the final dataframe contains these columns (transform should have created them)
    missing_required_columns = [col for col in ['StudentTeacherRatio', 'AvgScore'] if col not in df.columns]
    if missing_required_columns:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing_required_columns}")

    # Keep only predictors that exist as columns (transform ensures columns exist, but be defensive)
    predictors = [p for p in predictors if p in df.columns]

    if 'StudentTeacherRatio' not in predictors:
        # This should not happen if transform followed contract; provide helpful error
        raise ValueError("Required predictor 'StudentTeacherRatio' is missing from the dataframe.")

    X = df[predictors]
    X = sm.add_constant(X, has_constant='add')
    y = df['AvgScore']

    # Ensure there are observations after removing rows with missing endog/exog
    # We simulate statsmodels' missing='drop' behavior to count usable observations
    combined = pd.concat([y, X], axis=1)
    combined = combined.dropna(how='any')  # drop rows with any missing values used in the regression
    if combined.shape[0] == 0:
        raise ValueError("No observations available with non-missing outcome and predictors for estimation.")
    if combined.shape[0] < 2 and X.shape[1] > 1:
        # With very few observations, estimation will either fail or be uninformative;
        # allow statsmodels to run but warn via error to avoid confusing internal failures.
        raise ValueError(
            f"Too few observations ({combined.shape[0]}) after dropping missing values to reliably fit the model."
        )

    # Fit OLS with heteroskedasticity-robust (HC3) standard errors
    model_sm = sm.OLS(y, X, missing='drop')
    results = model_sm.fit(cov_type='HC3')

    return results