from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/add_features_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district dataframe into a cleaned dataframe containing the
    dependent variable (AvgScore), independent variable (StudentTeacherRatio),
    and control variables used in the regression.

    Final columns produced/kept:
      - AvgScore
      - StudentTeacherRatio
      - ExpenditurePerStudent
      - PctReducedLunch
      - PctEnglishLearners
      - ComputersPerStudent
      - IncomeThousands
      - GradeSpan_KK08
      - county

    Rows with missing values in any of these columns are dropped.
    """
    # Work on a copy
    df = df.copy()

    # Ensure expected columns exist; if not, this will raise a KeyError which notifies user
    required_raw_cols = ['students', 'teachers', 'read', 'math', 'expenditure', 'lunch', 'english', 'computer', 'income', 'grades', 'county']

    # Normalize column names (in case of stray whitespace) for string columns
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype(str).str.strip()
    if 'county' in df.columns:
        df['county'] = df['county'].astype(str).str.strip()

    # Drop rows with missing data in variables required for model
    df = df.dropna(subset=required_raw_cols)

    # Convert numeric columns to numeric types (coerce any non-numeric to NaN then drop)
    num_cols = ['students', 'teachers', 'read', 'math', 'expenditure', 'lunch', 'english', 'computer', 'income']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=num_cols)

    # Prevent division by zero (teachers should be > 0). If zero or very small, mark as NaN and drop.
    df.loc[df['teachers'] <= 0, 'teachers'] = np.nan
    df = df.dropna(subset=['teachers'])

    # Dependent variable: average of reading and math
    df['AvgScore'] = (df['read'] + df['math']) / 2.0

    # Independent variable: student-teacher ratio
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Controls: rename / derive to final column names used in model
    df['ExpenditurePerStudent'] = df['expenditure']
    df['PctReducedLunch'] = df['lunch']
    df['PctEnglishLearners'] = df['english']
    # Computers per student (if students > 0). If zero students (unlikely), result will be inf -> set to NaN
    df['ComputersPerStudent'] = df['computer'] / df['students']
    df.loc[~np.isfinite(df['ComputersPerStudent']), 'ComputersPerStudent'] = np.nan

    # Income is already in thousands per schema; keep name matching the model
    df['IncomeThousands'] = df['income']

    # Binary indicator for grade span KK-08 (1 if KK-08, else 0). Use string match after stripping.
    df['GradeSpan_KK08'] = df['grades'].fillna('').astype(str).str.strip().apply(lambda x: 1 if x == 'KK-08' else 0)

    # Ensure county is a clean categorical string (used as fixed effects)
    df['county'] = df['county'].astype(str).str.strip()

    # Finally select and return only the columns needed for the statistical model
    final_cols = [
        'AvgScore',
        'StudentTeacherRatio',
        'ExpenditurePerStudent',
        'PctReducedLunch',
        'PctEnglishLearners',
        'ComputersPerStudent',
        'IncomeThousands',
        'GradeSpan_KK08',
        'county'
    ]

    df_final = df[final_cols].dropna()  # drop any rows with missing values in final predictors/controls

    # Reset index for cleanliness
    df_final = df_final.reset_index(drop=True)

    return df_final


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression of AvgScore on StudentTeacherRatio controlling for district-level covariates
    and county fixed effects. Returns the fitted statsmodels results object with robust standard errors.

    Model specification:
      AvgScore ~ StudentTeacherRatio + ExpenditurePerStudent + PctReducedLunch +
                 PctEnglishLearners + ComputersPerStudent + IncomeThousands + GradeSpan_KK08 + C(county)

    Uses HC3 heteroskedasticity-robust standard errors.
    """
    import statsmodels.formula.api as smf

    # Ensure the necessary columns exist in the provided dataframe
    needed = ['AvgScore', 'StudentTeacherRatio', 'ExpenditurePerStudent', 'PctReducedLunch',
              'PctEnglishLearners', 'ComputersPerStudent', 'IncomeThousands', 'GradeSpan_KK08', 'county']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula with county as categorical fixed effect
    formula = (
        'AvgScore ~ StudentTeacherRatio + ExpenditurePerStudent + PctReducedLunch + '
        'PctEnglishLearners + ComputersPerStudent + IncomeThousands + GradeSpan_KK08 + C(county)'
    )

    # Fit OLS
    ols_model = smf.ols(formula=formula, data=df)
    results = ols_model.fit(cov_type='HC3')

    # Return the fitted results object. Users can call results.summary() for a printable summary.
    return results


