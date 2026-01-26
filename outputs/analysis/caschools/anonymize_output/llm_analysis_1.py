from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/anonymize_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the dataframe used by the statistical model.

    Creates these key columns (exact names used in modeling):
      - AvgTestScore: average of feature14 (avg reading) and feature15 (avg math)
      - StudentTeacherRatio: feature6 (enrollment) / feature7 (num teachers)
      - LogStudentTeacherRatio: natural log of StudentTeacherRatio
      - Enrollment, NumTeachers, Computers, ComputersPerStudent, ExpenditurePerStudent,
        PctCalWorks, PctReducedLunch, PctEnglishLearners, DistrictIncomeK,
        County, GradeSpan, LogEnrollment

    Drops rows with missing or invalid values for critical variables.
    """
    # copy to avoid modifying input in-place
    df = df.copy()

    # Ensure required columns exist. If not, this will raise a KeyError so the user can check schema.
    required_raw = ['feature6', 'feature7', 'feature14', 'feature15']
    # drop rows missing the core numeric fields needed to compute outcome and IV
    df = df.dropna(subset=required_raw)

    # Rename / create clearer columns from the raw feature names (keep raw columns intact in case user needs them)
    df['Enrollment'] = df['feature6'].astype(float)
    df['NumTeachers'] = df['feature7'].astype(float)
    df['Computers'] = df['feature10'].astype(float) if 'feature10' in df.columns else np.nan
    df['ExpenditurePerStudent'] = df['feature11'].astype(float) if 'feature11' in df.columns else np.nan
    df['DistrictIncomeK'] = df['feature12'].astype(float) if 'feature12' in df.columns else np.nan

    # Socioeconomic / demographic percentages
    df['PctCalWorks'] = df['feature8'].astype(float) if 'feature8' in df.columns else np.nan
    df['PctReducedLunch'] = df['feature9'].astype(float) if 'feature9' in df.columns else np.nan
    df['PctEnglishLearners'] = df['feature13'].astype(float) if 'feature13' in df.columns else np.nan

    # Categorical controls
    if 'feature4' in df.columns:
        df['County'] = df['feature4'].astype('category')
    else:
        df['County'] = pd.Categorical([None] * len(df))
    if 'feature5' in df.columns:
        df['GradeSpan'] = df['feature5'].astype('category')
    else:
        df['GradeSpan'] = pd.Categorical([None] * len(df))

    # Outcome: average of reading and math scores
    df['AvgTestScore'] = (df['feature14'].astype(float) + df['feature15'].astype(float)) / 2.0

    # Remove implausible / invalid values: teachers must be > 0
    df = df[df['NumTeachers'] > 0]

    # Independent variable: student-to-teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['Enrollment'] / df['NumTeachers']

    # Log transform to reduce right skew; keep both versions
    # Protect against non-positive values (shouldn't happen given NumTeachers >0 and Enrollment>0)
    df['LogStudentTeacherRatio'] = np.log(df['StudentTeacherRatio'].replace(0, np.nan))

    # Additional controls derived
    # Computers per student (if Enrollment is zero this will produce inf; handle by replacing inf with NaN)
    df['ComputersPerStudent'] = df['Computers'] / df['Enrollment']
    df.loc[~np.isfinite(df['ComputersPerStudent']), 'ComputersPerStudent'] = np.nan

    # Log Enrollment for scale control
    df['LogEnrollment'] = np.log(df['Enrollment'].replace(0, np.nan))

    # Keep only the columns needed for modeling (but we return full df with these columns added)
    model_cols = [
        'AvgTestScore', 'StudentTeacherRatio', 'LogStudentTeacherRatio',
        'Enrollment', 'LogEnrollment', 'NumTeachers', 'Computers', 'ComputersPerStudent',
        'ExpenditurePerStudent', 'PctCalWorks', 'PctReducedLunch', 'PctEnglishLearners',
        'DistrictIncomeK', 'County', 'GradeSpan'
    ]

    # It's fine if some of these are NaN; model function will drop missing rows used in estimation.
    # Return the augmented dataframe
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model estimating the relationship between student-teacher ratio and average test score,
    controlling for district characteristics and including county and grade-span fixed effects.

    Returns the fitted statsmodels regression result object (with robust HC3 standard errors).
    """
    import statsmodels.formula.api as smf

    # Work on a copy
    df = df.copy()

    # Define the list of columns required for the model and drop rows with missing values on these
    required_for_model = [
        'AvgTestScore', 'StudentTeacherRatio', 'ExpenditurePerStudent', 'ComputersPerStudent',
        'PctCalWorks', 'PctReducedLunch', 'PctEnglishLearners', 'DistrictIncomeK', 'LogEnrollment',
        'County', 'GradeSpan'
    ]
    # Drop rows missing any required variable
    model_df = df.dropna(subset=required_for_model)

    # Specify formula. County and GradeSpan entered as categorical fixed effects.
    # We use StudentTeacherRatio (linear) as the primary IV. Alternative specifications (e.g., log ratio)
    # can be run separately by replacing StudentTeacherRatio with LogStudentTeacherRatio.
    formula = (
        'AvgTestScore ~ StudentTeacherRatio '
        '+ ExpenditurePerStudent + ComputersPerStudent '
        '+ PctCalWorks + PctReducedLunch + PctEnglishLearners + DistrictIncomeK '
        '+ LogEnrollment'
        ' + C(County) + C(GradeSpan)'
    )

    # Fit OLS with robust (HC3) standard errors
    model = smf.ols(formula, data=model_df).fit(cov_type='HC3')

    # Return the fitted model object (user can call .summary() or access params, bse, etc.)
    return model


