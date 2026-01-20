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
    # Work on a copy
    df = df.copy()

    # Rename source columns for clarity (leave originals intact)
    df['Enrollment'] = df['feature6']
    df['NumTeachers'] = df['feature7']
    df['PctCalWorks'] = df['feature8']
    df['PctReducedPriceLunch'] = df['feature9']
    df['NumComputers'] = df['feature10']
    df['ExpenditurePerStudent'] = df['feature11']
    df['AvgIncomeK'] = df['feature12']
    df['PctEnglishLearners'] = df['feature13']
    df['AvgReading'] = df['feature14']
    df['AvgMath'] = df['feature15']
    df['GradeSpan'] = df['feature5']

    # Drop rows with missing or invalid teacher counts or enrollment
    df = df.dropna(subset=['Enrollment', 'NumTeachers', 'AvgReading', 'AvgMath'])
    # Exclude non-positive teacher counts to avoid division-by-zero or nonsensical ratios
    df = df[df['NumTeachers'] > 0]

    # Compute student-teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['Enrollment'] / df['NumTeachers']

    # Compute outcome: average of reading and math
    df['AvgScore'] = df[['AvgReading', 'AvgMath']].mean(axis=1)

    # Keep relevant numeric controls and standardize (z-score) them to aid interpretation and avoid scaling issues
    numeric_controls = [
        ('PctCalWorks', 'PctCalWorks_z'),
        ('PctReducedPriceLunch', 'PctReducedPriceLunch_z'),
        ('ExpenditurePerStudent', 'ExpenditurePerStudent_z'),
        ('AvgIncomeK', 'AvgIncomeK_z'),
        ('PctEnglishLearners', 'PctEnglishLearners_z'),
        ('NumComputers', 'NumComputers_z'),
        ('StudentTeacherRatio', 'StudentTeacherRatio_z')
    ]

    for orig, zname in numeric_controls:
        # If column is missing entirely, create NaNs to preserve shape
        if orig not in df.columns:
            df[zname] = np.nan
            continue
        col = df[orig]
        # Compute z-score with ddof=0 (population) but ddof=1 would also be acceptable
        mean = col.mean()
        std = col.std(ddof=0)
        # If std is zero (constant column), set z to 0 to avoid division by zero
        if pd.isna(std) or std == 0:
            df[zname] = 0.0
        else:
            df[zname] = (col - mean) / std

    # Encode GradeSpan as a single dummy (drop_first=True): create a column for 'KK-08' vs baseline 'KK-06' (or other)
    # This produces a column named like 'GradeSpan_KK-08' when that category exists.
    if 'GradeSpan' in df.columns:
        dummies = pd.get_dummies(df['GradeSpan'].astype(str), prefix='GradeSpan', drop_first=True)
        # If the expected dummy name exists (e.g., GradeSpan_KK-08) keep it; otherwise keep whatever dummy cols were generated.
        for col in dummies.columns:
            df[col] = dummies[col]
    else:
        # Ensure the model code can reference GradeSpan_KK-08 even if missing by creating column of zeros
        df['GradeSpan_KK-08'] = 0

    # Ensure the specific dummy used in the model exists (if not, create a zero column)
    if 'GradeSpan_KK-08' not in df.columns:
        df['GradeSpan_KK-08'] = 0

    # Final drop: remove rows with missing values in the model columns we will use
    required_cols = [
        'AvgScore',
        'StudentTeacherRatio_z',
        'PctCalWorks_z',
        'PctReducedPriceLunch_z',
        'ExpenditurePerStudent_z',
        'AvgIncomeK_z',
        'PctEnglishLearners_z',
        'NumComputers_z',
        'GradeSpan_KK-08'
    ]
    df = df.dropna(subset=required_cols)

    # Return the transformed dataframe containing all columns needed for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # Define predictor columns to use in the regression
    X_cols = [
        'StudentTeacherRatio_z',
        'PctCalWorks_z',
        'PctReducedPriceLunch_z',
        'ExpenditurePerStudent_z',
        'AvgIncomeK_z',
        'PctEnglishLearners_z',
        'NumComputers_z',
        'GradeSpan_KK-08'
    ]

    # Ensure columns exist in dataframe
    missing = [c for c in X_cols + ['AvgScore'] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for modeling: {missing}")

    X = df[X_cols]
    X = sm.add_constant(X)
    y = df['AvgScore']

    # Fit OLS regression: AvgScore ~ StudentTeacherRatio + controls
    model = sm.OLS(y, X).fit()

    # Return the fitted model results (RegressionResultsWrapper). The caller can inspect .summary() or coefficients.
    return model


