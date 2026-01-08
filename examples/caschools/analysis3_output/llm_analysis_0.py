from typing import Any
import pandas as pd
import statsmodels.api as sm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district-level dataframe into a modeling-ready dataframe.

    Steps:
    - Work on a copy.
    - Coerce relevant columns to numeric where appropriate.
    - Require enrollment ('calworks') and number of teachers ('teachers').
    - Allow either or both test-score columns ('grades' and 'rownames'); compute AvgTestScore
      as the row-wise mean of the available score columns (skipna=True).
    - Remove rows with nonpositive teacher counts.
    - Compute StudentTeacherRatio = calworks / teachers.
    - Keep only rows that have the final model columns present and non-missing.
    - Ensure 'school' exists as a categorical variable (placeholder 'unknown' if absent).
    - Return dataframe with the required final columns:
      ['StudentTeacherRatio', 'AvgTestScore', 'expenditure', 'math', 'district', 'english', 'school']
      (only those that are present in the input will be retained, but StudentTeacherRatio and AvgTestScore
       will always be produced when possible).
    """
    df = df.copy()

    # Coerce likely numeric columns to numeric where they exist
    numeric_cols = ['calworks', 'teachers', 'grades', 'rownames', 'expenditure', 'math', 'district', 'english']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure 'school' exists as categorical control; create placeholder if missing
    if 'school' in df.columns:
        df['school'] = df['school'].astype('category')
    else:
        df['school'] = pd.Categorical(['unknown'] * len(df))

    # Require enrollment and teachers to compute StudentTeacherRatio
    required_for_ratio = [c for c in ['calworks', 'teachers'] if c in df.columns]
    if not required_for_ratio:
        # Can't compute ratio without these columns; return empty dataframe with appropriate columns
        return df.reset_index(drop=True)

    df = df.dropna(subset=required_for_ratio)

    # Remove implausible teacher counts (zero or negative)
    if 'teachers' in df.columns:
        df = df[df['teachers'] > 0]

    # Identify available score columns and require at least one to compute AvgTestScore
    score_cols = [c for c in ['grades', 'rownames'] if c in df.columns]
    if not score_cols:
        # No score columns available; return empty dataframe (nothing to model)
        return df.reset_index(drop=True)

    # Compute StudentTeacherRatio
    df['StudentTeacherRatio'] = df['calworks'] / df['teachers']

    # Compute AvgTestScore as mean of available score columns (skip missing)
    df['AvgTestScore'] = df[score_cols].mean(axis=1, skipna=True)

    # Remove rows where both score columns were missing (mean will be NaN) or StudentTeacherRatio is NaN
    df = df.dropna(subset=['StudentTeacherRatio', 'AvgTestScore'])

    # Finally, ensure the final model columns exist and are non-missing for retained rows
    model_columns = ['StudentTeacherRatio', 'AvgTestScore', 'expenditure', 'math', 'district', 'english', 'school']
    model_columns_present = [c for c in model_columns if c in df.columns]
    df = df.dropna(subset=[c for c in model_columns_present if c not in ['school']])

    # Ensure 'school' is categorical
    if 'school' in df.columns:
        df['school'] = df['school'].astype('category')

    df = df.reset_index(drop=True)
    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression predicting AvgTestScore from StudentTeacherRatio and controls.

    Model specification:
    AvgTestScore ~ StudentTeacherRatio + expenditure + math + district + english + categorical school fixed effects

    Returns the fitted statsmodels regression results object.
    """
    df = df.copy()

    # Ensure required columns are present
    required = ['StudentTeacherRatio', 'AvgTestScore']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in dataframe")

    # Drop rows with missing outcome or key regressor
    df = df.dropna(subset=['StudentTeacherRatio', 'AvgTestScore']).reset_index(drop=True)

    if df.shape[0] == 0:
        raise ValueError("No observations available to fit the model after dropna in model()")

    # Base predictors
    predictors = ['StudentTeacherRatio']
    controls = [c for c in ['expenditure', 'math', 'district', 'english'] if c in df.columns]
    predictors += controls

    # Build X dataframe from predictors (these are the conceptual variable columns)
    X = df[predictors].astype(float).copy()

    # Add school dummies if 'school' exists and has more than one category
    if 'school' in df.columns:
        # Ensure categorical dtype
        df['school'] = df['school'].astype('category')
        # If only one category, get_dummies with drop_first will produce empty df; that's okay
        school_dummies = pd.get_dummies(df['school'], prefix='school', drop_first=True, dtype=float)
        if not school_dummies.empty:
            X = pd.concat([X, school_dummies], axis=1)

    # Ensure there is at least one regressor column before adding constant
    if X.shape[1] == 0:
        raise ValueError("No regressors available to fit the model (after processing controls and school dummies)")

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Outcome variable
    y = df['AvgTestScore'].astype(float)

    # Fit OLS; statsmodels expects non-empty arrays
    if X.shape[0] == 0:
        raise ValueError("No observations available to fit the model")

    results = sm.OLS(y, X).fit()
    return results