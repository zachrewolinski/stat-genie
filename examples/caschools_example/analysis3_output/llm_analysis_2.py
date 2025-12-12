from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Helper: find first existing column in dataframe from a list of candidates
def _find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a working copy
    df = df.copy()

    # Normalize column names access by keeping original but searching for likely candidates.
    # Enrollment
    enrollment_col = _find_col(df, ['Enrollment', 'enrollment', 'calworks', 'enroll', 'enrol', 'enrollment_total'])
    if enrollment_col is not None:
        df['Enrollment'] = pd.to_numeric(df[enrollment_col], errors='coerce')
    else:
        df['Enrollment'] = np.nan

    # Teachers (full-time-equivalent teacher count)
    teachers_col = _find_col(df, ['teachers', 'teacher_count', 'num_teachers', 'Teachers', 'TeacherFTE', 'teacher_fte'])
    if teachers_col is not None:
        df['teachers'] = pd.to_numeric(df[teachers_col], errors='coerce')
    else:
        # Ensure column exists for later computation; it may be NaN and rows will be handled downstream
        df['teachers'] = np.nan

    # Expenditure per student
    expend_col = _find_col(df, ['expenditure', 'expend', 'spending_per_student', 'expend_per_student', 'expenditure_per_student', 'expend_per_student'])
    if expend_col is not None:
        df['expenditure'] = pd.to_numeric(df[expend_col], errors='coerce')
    else:
        df['expenditure'] = np.nan

    # Percent reduced-price lunch (socioeconomic proxy)
    prl_col = _find_col(df, ['PctReducedLunch', 'pct_reduced_lunch', 'frl', 'free_reduced_lunch', 'reducedprice', 'reduced_price_lunch', 'pctfrl', 'percent_reduced_lunch'])
    if prl_col is not None:
        df['PctReducedLunch'] = pd.to_numeric(df[prl_col], errors='coerce')
    else:
        df['PctReducedLunch'] = np.nan

    # Percent English learners
    ell_col = _find_col(df, ['PctEnglishLearners', 'pct_english_learners', 'ell', 'english_learners', 'pctell', 'pct_ell', 'Pct_EL', 'ELL'])
    if ell_col is not None:
        df['PctEnglishLearners'] = pd.to_numeric(df[ell_col], errors='coerce')
    else:
        df['PctEnglishLearners'] = np.nan

    # Computers per classroom / computers indicator
    comp_col = _find_col(df, ['ComputersPerClassroom', 'computers_per_classroom', 'computer', 'computers', 'comp', 'computers_per_room'])
    if comp_col is not None:
        df['ComputersPerClassroom'] = pd.to_numeric(df[comp_col], errors='coerce')
    else:
        df['ComputersPerClassroom'] = np.nan

    # Scores: try to find reading and math columns; compute AvgScore as mean of available score columns
    read_col = _find_col(df, ['reading', 'read', 'reading_score', 'read_score', 'reading_mean', 'reading_avg'])
    math_col = _find_col(df, ['math', 'math_score', 'math_mean', 'math_avg'])
    score_cols = []
    if read_col is not None:
        df[read_col] = pd.to_numeric(df[read_col], errors='coerce')
        score_cols.append(read_col)
    if math_col is not None:
        df[math_col] = pd.to_numeric(df[math_col], errors='coerce')
        score_cols.append(math_col)
    # If no typical columns found, try any numeric columns that look like scores
    if not score_cols:
        for c in df.columns:
            if c.lower() in ('score', 'avgscore', 'avg_score', 'average_score', 'mean_score'):
                df[c] = pd.to_numeric(df[c], errors='coerce')
                score_cols.append(c)
                break

    if score_cols:
        df['AvgScore'] = df[score_cols].mean(axis=1)
    else:
        df['AvgScore'] = np.nan

    # Student-Teacher Ratio: try to use existing ratio if present; else compute Enrollment / teachers
    stratio_col = _find_col(df, ['StudentTeacherRatio', 'student_teacher_ratio', 'studentteacher', 'student_teacher', 'stratio'])
    if stratio_col is not None:
        df['StudentTeacherRatio'] = pd.to_numeric(df[stratio_col], errors='coerce')
    else:
        # Ensure numeric types
        df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce')
        df['teachers'] = pd.to_numeric(df['teachers'], errors='coerce')
        # Prevent division by zero
        df.loc[df['teachers'] == 0, 'teachers'] = np.nan
        df['StudentTeacherRatio'] = df['Enrollment'] / df['teachers']

    # school: grade-span type. Prefer explicit 'school' column; else use 'grades' if present.
    if 'school' in df.columns and not df['school'].isnull().all():
        df['school'] = df['school'].astype(object)
    else:
        # Use 'grades' as fallback for grade-span if available
        if 'grades' in df.columns:
            df['school'] = df['grades'].astype(object)
        else:
            # Ensure the column exists (may be all NaN) with proper index
            df['school'] = pd.Series(np.nan, index=df.index)

    # Ensure final required columns exist exactly as specified.
    required_final = [
        'StudentTeacherRatio',
        'AvgScore',
        'expenditure',
        'PctReducedLunch',
        'PctEnglishLearners',
        'ComputersPerClassroom',
        'Enrollment',
        'school'
    ]
    for col in required_final:
        if col not in df.columns:
            df[col] = np.nan

    # Cast numeric columns to numeric dtype to avoid object dtypes hidden
    numeric_cols = ['StudentTeacherRatio', 'AvgScore', 'expenditure', 'PctReducedLunch', 'PctEnglishLearners', 'ComputersPerClassroom', 'Enrollment']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def model(df: pd.DataFrame) -> Any:
    # Working copy
    df = df.copy()

    # Required model columns
    model_cols = ['AvgScore', 'StudentTeacherRatio', 'expenditure', 'PctReducedLunch', 'PctEnglishLearners', 'ComputersPerClassroom', 'Enrollment', 'school']

    # Ensure model columns exist
    missing = [c for c in model_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing needed columns for modeling: {missing}")

    # Drop rows with missing outcome or primary independent variable
    df_model = df.dropna(subset=['AvgScore', 'StudentTeacherRatio']).copy()

    # If no observations after dropping outcome/primary IV, nothing to do
    if df_model.shape[0] == 0:
        raise ValueError("No observations available for modeling after dropping missing AvgScore or StudentTeacherRatio.")

    # Prepare controls: impute missing control values rather than dropping rows to preserve observations
    controls = ['expenditure', 'PctReducedLunch', 'PctEnglishLearners', 'ComputersPerClassroom', 'Enrollment']
    # Ensure numeric dtype for controls
    for c in controls:
        df_model[c] = pd.to_numeric(df_model[c], errors='coerce')

    # Impute numeric controls with median where possible; if median is NaN (all missing), fill with 0
    for c in controls:
        median_val = df_model[c].median(skipna=True)
        if pd.isna(median_val):
            # If there is no information to impute, fill with 0 to allow model to run.
            df_model[c] = df_model[c].fillna(0)
        else:
            df_model[c] = df_model[c].fillna(median_val)

    # Create design matrix X with the exact columns required
    X = df_model[['StudentTeacherRatio', 'expenditure', 'PctReducedLunch', 'PctEnglishLearners', 'ComputersPerClassroom', 'Enrollment']].copy()

    # Ensure numeric dtype for all X columns
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')

    # After imputation, if any remaining NaNs exist in predictors, drop those rows
    non_na_mask = ~X.isnull().any(axis=1)
    if non_na_mask.sum() == 0:
        raise ValueError("No predictor observations available after imputation and removing rows with remaining NaNs.")
    X = X.loc[non_na_mask].copy()
    y = pd.to_numeric(df_model.loc[non_na_mask, 'AvgScore'], errors='coerce')

    # Encode 'school' categorical variable using one-hot dummies only if it has at least 2 non-null unique levels
    school_series = df_model.loc[non_na_mask, 'school'].astype(object)
    unique_levels = school_series.dropna().unique()
    if len(unique_levels) >= 2:
        dummies = pd.get_dummies(school_series, prefix='school', drop_first=True)
        # Ensure dummies are numeric
        for c in dummies.columns:
            dummies[c] = pd.to_numeric(dummies[c], errors='coerce')
        if not dummies.empty:
            X = pd.concat([X, dummies], axis=1)

    # Add constant/intercept
    X = sm.add_constant(X, has_constant='add')

    # Final sanity checks
    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("No observations available for modeling after preparing X and y.")
    if X.shape[0] != y.shape[0]:
        # Align indices just in case
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        if X.shape[0] == 0:
            raise ValueError("Mismatch in number of observations between X and y after alignment.")

    # Fit OLS model
    model_res = sm.OLS(y, X).fit()

    # Print brief summary and return results
    print(model_res.summary())
    return model_res