from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/caschools/noperturb_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district-level dataframe into analysis-ready variables.

    Produces the following columns required for modeling:
    - StudentTeacherRatio: students / teachers
    - AvgScore: mean of 'read' and 'math'
    - Expenditure, Income, PercentLunch, CalWorks, PercentEnglishLearners (mapped from original cols)
    - ComputersPer100Students: computer / students * 100
    - LogStudents: natural log of students
    - Grades_KK08: binary indicator for 'KK-08'
    - County: copied from original 'county' column (categorical)

    Rows with missing values on the variables needed for the model are dropped.
    """
    df = df.copy()

    # Ensure required raw columns exist; rename to consistent internal names where useful
    required_raw = ['students', 'teachers', 'read', 'math', 'expenditure', 'income', 'lunch', 'calworks', 'english', 'computer', 'grades', 'county']
    missing_cols = [c for c in required_raw if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in input dataframe: {missing_cols}")

    # Drop rows with missing core values (students, teachers, scores)
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove unrealistic/nonpositive teacher counts to avoid division issues
    df = df[df['teachers'] > 0]

    # Create student-teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Map and copy control variables to canonical column names used in modeling
    df['Expenditure'] = df['expenditure']
    df['Income'] = df['income']
    df['PercentLunch'] = df['lunch']
    df['CalWorks'] = df['calworks']
    df['PercentEnglishLearners'] = df['english']

    # Computers per 100 students (handle zero students if any by producing NaN)
    df['ComputersPer100Students'] = df['computer'] / df['students'] * 100

    # Log of student enrollment (add small constant if there were zeros, but we dropped zero teachers above; ensure positive)
    df['LogStudents'] = np.log(df['students'].astype(float))

    # Binary indicator for grade span KK-08 (1 if KK-08, else 0)
    df['Grades_KK08'] = df['grades'].apply(lambda x: 1 if str(x).strip() == 'KK-08' else 0)

    # County as categorical control (keep original values)
    df['County'] = df['county']

    # Final required columns for the regression
    final_cols = [
        'StudentTeacherRatio', 'AvgScore', 'Expenditure', 'Income', 'PercentLunch', 'CalWorks',
        'PercentEnglishLearners', 'ComputersPer100Students', 'LogStudents', 'Grades_KK08', 'County'
    ]

    # Drop rows with any missing values among final columns
    df = df.dropna(subset=final_cols)

    # Return only original columns plus the created columns (keeps dataset manageable)
    keep_cols = list(df.columns.intersection(required_raw)) + final_cols
    # Preserve index and other metadata by returning full df (but ensure final columns exist)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression to estimate the association between student-teacher ratio and average test scores,
    controlling for district resources and demographics. County fixed effects are included via categorical
    dummies. Robust (HC3) standard errors are used.

    Returns the fitted statsmodels RegressionResults object.
    """
    import statsmodels.formula.api as smf

    # Ensure required model columns are present
    model_vars = [
        'AvgScore', 'StudentTeacherRatio', 'Expenditure', 'Income', 'PercentLunch', 'CalWorks',
        'PercentEnglishLearners', 'ComputersPer100Students', 'LogStudents', 'Grades_KK08', 'County'
    ]
    missing = [v for v in model_vars if v not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for model: {missing}")

    # Specify formula: include county fixed effects via C(County)
    formula = (
        'AvgScore ~ StudentTeacherRatio + Expenditure + Income + PercentLunch + CalWorks + '
        'PercentEnglishLearners + ComputersPer100Students + LogStudents + Grades_KK08 + C(County)'
    )

    # Fit OLS with heteroskedasticity-robust standard errors (HC3)
    model_res = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    # Return the fitted model object (contains params, summary, etc.)
    return model_res


