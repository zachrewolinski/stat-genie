from typing import Any, List, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe suitable for modeling.

    Produces the derived columns used in the model:
    - Enrollment (feature6)
    - NumTeachers (feature7)
    - NumComputers (feature10)
    - ExpenditurePerStudent (feature11)
    - DistrictIncomeK (feature12)
    - PctEnglishLearners (feature13)
    - AvgReading (feature14)
    - AvgMath (feature15)
    - PctReducedLunch (feature9)
    - StudentTeacherRatio = Enrollment / NumTeachers
    - AvgScore = mean(AvgReading, AvgMath)
    - ComputersPerStudent = NumComputers / Enrollment

    The function is made robust to common alternative column namings by attempting to
    find matching columns by keyword if the exact 'featureX' columns are not present.
    """
    df = df.copy()

    def find_first_match(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
        """Return the first column name in df that contains any of the keywords (case-insensitive)."""
        lower_cols = {c: c.lower() for c in df.columns}
        for kw in keywords:
            for orig_col, lower_col in lower_cols.items():
                if kw in lower_col:
                    return orig_col
        return None

    def get_series(df: pd.DataFrame, primary: str, alt_keywords: List[str]) -> pd.Series:
        """
        Return a numeric series from df using primary column name if present,
        otherwise try to find a column matching any of alt_keywords. If nothing found,
        return a numeric series of NaNs with the same index as df.
        """
        if primary in df.columns:
            s = df[primary]
        else:
            match = find_first_match(df, alt_keywords)
            if match:
                s = df[match]
            else:
                s = pd.Series([np.nan] * len(df), index=df.index)
        return pd.to_numeric(s, errors='coerce')

    # Map original features to clearer column names, with fallbacks for common alternative names
    df['Enrollment'] = get_series(df, 'feature6', ['enroll', 'student', 'students', 'enrollment'])
    df['NumTeachers'] = get_series(df, 'feature7', ['teacher', 'teachers', 'fte'])
    df['NumComputers'] = get_series(df, 'feature10', ['computer', 'computers', 'pc', 'device'])
    df['ExpenditurePerStudent'] = get_series(df, 'feature11', ['expend', 'expenditure', 'spend', 'spending', 'per student'])
    df['DistrictIncomeK'] = get_series(df, 'feature12', ['income', 'district income', 'median income', 'incomek'])
    df['PctEnglishLearners'] = get_series(df, 'feature13', ['ell', 'english learn', 'english_learner', 'englishlearner', 'english'])
    df['AvgReading'] = get_series(df, 'feature14', ['read', 'reading'])
    df['AvgMath'] = get_series(df, 'feature15', ['math', 'mathemat'])
    df['PctReducedLunch'] = get_series(df, 'feature9', ['lunch', 'reduced', 'free lunch', 'reduced-price', 'reducedprice'])

    # Derived variables computed safely (avoid division-by-zero warnings)
    # StudentTeacherRatio: Enrollment divided by NumTeachers when NumTeachers > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        df['StudentTeacherRatio'] = np.where(
            (df['NumTeachers'].notna()) & (df['NumTeachers'] != 0),
            df['Enrollment'] / df['NumTeachers'],
            np.nan
        )
        # AvgScore: mean of AvgReading and AvgMath (rowwise)
        df['AvgScore'] = df[['AvgReading', 'AvgMath']].mean(axis=1)
        # ComputersPerStudent: NumComputers divided by Enrollment when Enrollment > 0
        df['ComputersPerStudent'] = np.where(
            (df['Enrollment'].notna()) & (df['Enrollment'] != 0),
            df['NumComputers'] / df['Enrollment'],
            np.nan
        )

    # Ensure derived columns are numeric
    derived_cols = ['StudentTeacherRatio', 'AvgScore', 'ComputersPerStudent']
    for c in derived_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing any of the variables required for the model
    required = [
        'StudentTeacherRatio',
        'AvgScore',
        'ExpenditurePerStudent',
        'PctReducedLunch',
        'PctEnglishLearners',
        'DistrictIncomeK',
        'ComputersPerStudent'
    ]
    # Only drop using the subset of required columns that actually exist in df to avoid KeyError,
    # but ensure all required columns exist in final dataframe (they must, even if filled with NaN).
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    df = df.dropna(subset=required)

    # Winsorize/clip extreme StudentTeacherRatio values at 1st and 99th percentiles to reduce influence of outliers
    if not df['StudentTeacherRatio'].empty:
        lower = df['StudentTeacherRatio'].quantile(0.01)
        upper = df['StudentTeacherRatio'].quantile(0.99)
        if pd.notna(lower) and pd.notna(upper):
            df['StudentTeacherRatio'] = df['StudentTeacherRatio'].clip(lower, upper)

    # Return the dataframe with the new columns used by the model
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression of AvgScore on StudentTeacherRatio controlling for relevant covariates.

    Model specification:
      AvgScore ~ StudentTeacherRatio + ExpenditurePerStudent + PctReducedLunch + PctEnglishLearners + DistrictIncomeK + ComputersPerStudent

    Uses heteroskedasticity-robust (HC3) standard errors.

    Returns the fitted statsmodels regression results object.
    """
    df = df.copy()

    # Ensure required columns exist
    model_cols = [
        'AvgScore',
        'StudentTeacherRatio',
        'ExpenditurePerStudent',
        'PctReducedLunch',
        'PctEnglishLearners',
        'DistrictIncomeK',
        'ComputersPerStudent'
    ]
    missing = [c for c in model_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    df_model = df.dropna(subset=model_cols)

    if df_model.shape[0] == 0:
        raise ValueError("No observations available after dropping missing values required for the model.")

    # Prepare outcome and predictors
    y = df_model['AvgScore'].astype(float)
    X = df_model[['StudentTeacherRatio', 'ExpenditurePerStudent', 'PctReducedLunch', 'PctEnglishLearners', 'DistrictIncomeK', 'ComputersPerStudent']].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # Fit OLS with robust standard errors (HC3)
    results = sm.OLS(y, X).fit(cov_type='HC3')

    return results