from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/positive_leading_statement_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district dataset to produce the variables used in modeling.

    Outputs (columns created/kept):
      - StudentsPerTeacher: students / teachers, winsorized at 1st/99th pct.
      - StudentsPerTeacher_z: z-scored StudentsPerTeacher (mean 0, sd 1).
      - CompositeScore: (read + math) / 2.
      - LogTotalStudents: natural log of students (after clipping at minimum 1).
      - TeachersFTE: same as original 'teachers' (renamed for clarity).
      - income, calworks, lunch, computer, expenditure, english: kept from original.
      - grades_KK08: binary indicator 1 if grades == 'KK-08', else 0.
      - county, school, district kept for reference but not used directly in baseline model.
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['students', 'teachers', 'read', 'math']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    # Drop rows with missing essential measures
    df = df.dropna(subset=['students', 'teachers', 'read', 'math']).reset_index(drop=True)

    # Remove rows where teachers is zero or nearly zero to avoid infinite ratios
    df = df[df['teachers'] > 0].copy()

    # Standardize numeric column names we'll use directly
    df['TotalStudents'] = df['students'].astype(float)
    df['TeachersFTE'] = df['teachers'].astype(float)

    # Raw student-teacher ratio
    df['StudentsPerTeacher'] = df['TotalStudents'] / df['TeachersFTE']

    # Winsorize extreme ratios at 1st and 99th percentiles to reduce influence of outliers
    lower = df['StudentsPerTeacher'].quantile(0.01)
    upper = df['StudentsPerTeacher'].quantile(0.99)
    df['StudentsPerTeacher'] = df['StudentsPerTeacher'].clip(lower=lower, upper=upper)

    # Standardize (z-score) the student-teacher ratio for interpretability
    mean_spt = df['StudentsPerTeacher'].mean()
    std_spt = df['StudentsPerTeacher'].std(ddof=0) if df['StudentsPerTeacher'].std(ddof=0) > 0 else 1.0
    df['StudentsPerTeacher_z'] = (df['StudentsPerTeacher'] - mean_spt) / std_spt

    # Composite academic outcome: average of reading and math scores
    df['CompositeScore'] = (df['read'].astype(float) + df['math'].astype(float)) / 2.0

    # Log of total students (clip at 1 to avoid log(0))
    df['LogTotalStudents'] = np.log(df['TotalStudents'].clip(lower=1.0))

    # Keep relevant control variables (if they exist); otherwise fill with NaN to preserve schema
    for col in ['income', 'calworks', 'lunch', 'computer', 'expenditure', 'english']:
        if col in df.columns:
            # Ensure numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    # Grade-span binary indicator (KK-08 vs KK-06) if 'grades' present
    if 'grades' in df.columns:
        df['grades_KK08'] = df['grades'].astype(str).apply(lambda x: 1 if x.strip() == 'KK-08' else 0)
    else:
        df['grades_KK08'] = np.nan

    # Keep identifiers for reference
    for idcol in ['county', 'school', 'district']:
        if idcol not in df.columns:
            df[idcol] = np.nan

    # Final column ordering (helpful for readability)
    cols_order = [
        'district', 'school', 'county', 'grades', 'grades_KK08',
        'TotalStudents', 'TeachersFTE', 'StudentsPerTeacher', 'StudentsPerTeacher_z', 'LogTotalStudents',
        'CompositeScore',
        'income', 'calworks', 'lunch', 'computer', 'expenditure', 'english'
    ]
    # Append any other columns that may exist
    remaining = [c for c in df.columns if c not in cols_order]
    df = df[cols_order + remaining]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Runs an OLS regression to estimate the association between student-teacher ratio and academic performance.

    Model specification (baseline):
      CompositeScore ~ StudentsPerTeacher_z + income + calworks + lunch + computer + expenditure + english + LogTotalStudents + grades_KK08

    Uses heteroskedasticity-robust (HC3) standard errors.

    Returns:
      - results: fitted statsmodels regression results object (with robust cov).
    Also prints the main coefficient and a short summary.
    """
    df = df.copy()

    # Drop rows with missing outcome or key IV
    model_df = df.dropna(subset=['CompositeScore', 'StudentsPerTeacher_z']).copy()

    # Define predictors (controls). Keep rows with at least some non-missing control info; we will allow statsmodels to handle NaNs by dropping them.
    predictors = ['StudentsPerTeacher_z', 'income', 'calworks', 'lunch', 'computer', 'expenditure', 'english', 'LogTotalStudents', 'grades_KK08']

    # Ensure predictors exist in dataframe
    for p in predictors:
        if p not in model_df.columns:
            model_df[p] = np.nan

    # Drop rows with any NaN in predictors or outcome (listwise deletion for baseline model)
    model_df = model_df.dropna(subset=['CompositeScore'] + predictors).copy()

    if model_df.shape[0] < 20:
        raise ValueError('Too few complete cases to fit the model after dropping missing values.')

    # Design matrix
    X = model_df[predictors].astype(float)
    X = sm.add_constant(X)
    y = model_df['CompositeScore'].astype(float)

    # Fit OLS with robust standard errors (HC3)
    results = sm.OLS(y, X).fit(cov_type='HC3')

    # Print concise summary for the key IV
    spt_coef = results.params.get('StudentsPerTeacher_z', np.nan)
    spt_p = results.pvalues.get('StudentsPerTeacher_z', np.nan)
    spt_se = results.bse.get('StudentsPerTeacher_z', np.nan)

    print('Baseline OLS: CompositeScore ~ StudentsPerTeacher_z + controls')
    print(f'StudentsPerTeacher_z coefficient: {spt_coef:.4f} (SE={spt_se:.4f}), p={spt_p:.4f}')
    print('\nFull model summary:')
    print(results.summary())

    return results


