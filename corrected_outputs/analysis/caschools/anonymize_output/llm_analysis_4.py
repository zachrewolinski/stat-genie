from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/caschools/anonymize_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform original dataset into the analysis dataframe.

    Final columns produced (and used in modeling):
      - StudentTeacherRatio: enrollment / teachers
      - AvgTestScore: mean of feature14 (reading) and feature15 (math)
      - ExpenditurePerStudent: feature11
      - PercentReducedLunch: feature9
      - PercentCalWorks: feature8
      - PercentEnglishLearners: feature13
      - ComputersPerStudent: feature10 / feature6
      - AvgIncome: feature12
      - TotalEnrollment: feature6
      - GradeSpan_KK_08: dummy for feature5 ('KK-08' as 1, 'KK-06' as 0 after drop_first)

    Rows with missing values in the core variables are dropped.
    """
    df = df.copy()

    # Ensure numeric types for relevant numeric columns
    num_cols = ['feature6', 'feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14', 'feature15']
    for c in num_cols:
        # coerce errors to NaN so we can drop later
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Rename some source columns to clear names for intermediate use
    df.rename(columns={
        'feature6': 'TotalEnrollment_raw',
        'feature7': 'NumTeachers_raw',
        'feature8': 'PercentCalWorks',
        'feature9': 'PercentReducedLunch',
        'feature10': 'NumComputers',
        'feature11': 'ExpenditurePerStudent',
        'feature12': 'AvgIncome',
        'feature13': 'PercentEnglishLearners',
        'feature14': 'AvgReadingScore',
        'feature15': 'AvgMathScore',
        'feature5': 'GradeSpan'
    }, inplace=True)

    # Create dependent variable: average of reading and math scores
    df['AvgTestScore'] = df[['AvgReadingScore', 'AvgMathScore']].mean(axis=1)

    # Create Student-Teacher ratio (handle zero / missing teachers)
    df['TotalEnrollment'] = df['TotalEnrollment_raw']
    df['NumTeachers'] = df['NumTeachers_raw']
    df['StudentTeacherRatio'] = np.where(
        (df['NumTeachers'].notna()) & (df['NumTeachers'] > 0),
        df['TotalEnrollment'] / df['NumTeachers'],
        np.nan
    )

    # Computers per student (handle division by zero / missing)
    df['ComputersPerStudent'] = np.where(
        (df['TotalEnrollment'].notna()) & (df['TotalEnrollment'] > 0),
        df['NumComputers'] / df['TotalEnrollment'],
        np.nan
    )

    # Keep the expenditure and income columns as-is (already numeric-coerced)
    df['ExpenditurePerStudent'] = df['ExpenditurePerStudent']
    df['AvgIncome'] = df['AvgIncome']

    # Controls (ensure they exist)
    df['PercentReducedLunch'] = df['PercentReducedLunch']
    df['PercentCalWorks'] = df['PercentCalWorks']
    df['PercentEnglishLearners'] = df['PercentEnglishLearners']

    # Encode GradeSpan as dummy(s). Replace hyphens with underscores in dummy column names.
    df['GradeSpan'] = df['GradeSpan'].astype(str)
    grade_dummies = pd.get_dummies(df['GradeSpan'], prefix='GradeSpan', drop_first=True)
    # normalize column names to avoid '-' in names
    grade_dummies.columns = [c.replace('-', '_') for c in grade_dummies.columns]
    df = pd.concat([df, grade_dummies], axis=1)

    # Final selection: keep only rows with required variables non-missing
    required = [
        'StudentTeacherRatio', 'AvgTestScore', 'ExpenditurePerStudent', 'PercentReducedLunch',
        'PercentCalWorks', 'PercentEnglishLearners', 'ComputersPerStudent', 'AvgIncome', 'TotalEnrollment'
    ]
    # Also include grade dummy if present
    if 'GradeSpan_KK_08' in df.columns:
        required.append('GradeSpan_KK_08')
    # Drop rows with missing values in required columns
    df = df.dropna(subset=required)

    # Keep only the final columns needed for modeling (and helpful originals for inspection)
    final_cols = [
        'StudentTeacherRatio', 'AvgTestScore', 'ExpenditurePerStudent', 'PercentReducedLunch',
        'PercentCalWorks', 'PercentEnglishLearners', 'ComputersPerStudent', 'AvgIncome', 'TotalEnrollment'
    ]
    if 'GradeSpan_KK_08' in df.columns:
        final_cols.append('GradeSpan_KK_08')

    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model predicting AvgTestScore from StudentTeacherRatio and controls.

    Model specification:
      AvgTestScore = beta0 + beta1 * StudentTeacherRatio + sum(beta_k * controls_k) + eps

    Uses heteroskedasticity-robust (HC3) standard errors.

    Returns the fitted statsmodels results object.
    """
    # Ensure a copy
    df = df.copy()

    # Define outcome and predictors
    y = df['AvgTestScore']

    # Base set of controls
    X_cols = [
        'StudentTeacherRatio',
        'ExpenditurePerStudent',
        'PercentReducedLunch',
        'PercentCalWorks',
        'PercentEnglishLearners',
        'ComputersPerStudent',
        'AvgIncome',
        'TotalEnrollment'
    ]
    # Include grade span dummy if present
    if 'GradeSpan_KK_08' in df.columns:
        X_cols.append('GradeSpan_KK_08')

    X = df[X_cols]

    # Add constant for intercept
    X = sm.add_constant(X)

    # Fit OLS with robust standard errors (HC3)
    model = sm.OLS(y, X).fit(cov_type='HC3')

    return model


