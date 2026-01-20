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
    Transform the raw dataset to produce the columns required for modeling.

    Input columns (from provided schema):
      - feature6: Total enrollment
      - feature7: Number of teachers (FTE)
      - feature14: Average reading score
      - feature15: Average math score
      - feature8: Percent qualifying for CalWorks
      - feature9: Percent qualifying for reduced-price lunch
      - feature10: Number of computers
      - feature11: Expenditure per student
      - feature12: District average income (in 1,000s)
      - feature13: Percent English learners
      - feature5: Grade span (KK-06 or KK-08)
      - feature4: County (not used directly in base model but preserved)

    Returns:
      dataframe with new columns used in the model (see conceptual variables).
    """
    df = df.copy()

    # Rename relevant source columns to clearer names (preserve originals)
    df['TotalEnrollment'] = pd.to_numeric(df.get('feature6'), errors='coerce')
    df['NumTeachers'] = pd.to_numeric(df.get('feature7'), errors='coerce')

    # Compute student-teacher ratio; guard against zero or missing teachers
    df['StudentTeacherRatio'] = np.where(
        (df['NumTeachers'].notna()) & (df['NumTeachers'] > 0),
        df['TotalEnrollment'] / df['NumTeachers'],
        np.nan
    )

    # Dependent variable: average of reading and math scores
    df['ReadingScore'] = pd.to_numeric(df.get('feature14'), errors='coerce')
    df['MathScore'] = pd.to_numeric(df.get('feature15'), errors='coerce')
    df['AvgTestScore'] = df[['ReadingScore', 'MathScore']].mean(axis=1)

    # Controls: bring into clear columns and coerce to numeric where appropriate
    df['PctCalWorks'] = pd.to_numeric(df.get('feature8'), errors='coerce')
    df['PctReducedLunch'] = pd.to_numeric(df.get('feature9'), errors='coerce')
    df['NumComputers'] = pd.to_numeric(df.get('feature10'), errors='coerce')
    df['ExpenditurePerStudent'] = pd.to_numeric(df.get('feature11'), errors='coerce')
    df['DistrictIncomeK'] = pd.to_numeric(df.get('feature12'), errors='coerce')
    df['PctEnglishLearners'] = pd.to_numeric(df.get('feature13'), errors='coerce')

    # Categorical control: grade span -> create dummy for KK-08 (reference: KK-06)
    df['GradeSpan'] = df.get('feature5')
    df['GradeSpan_KK08'] = np.where(df['GradeSpan'].astype(str).str.strip() == 'KK-08', 1, 0)

    # Basic filtering: drop rows missing key variables (IV or DV)
    df = df.dropna(subset=['StudentTeacherRatio', 'AvgTestScore'])

    # Remove implausible ratios (e.g., extremely large > 1000) which likely indicate data issues
    df.loc[df['StudentTeacherRatio'] > 1000, 'StudentTeacherRatio'] = np.nan
    df = df.dropna(subset=['StudentTeacherRatio'])

    # Standardize continuous predictors (z-score) to make coefficients comparable
    def zscore(col):
        # ddof=0 for population sd; works with numeric series
        col = pd.to_numeric(col, errors='coerce')
        if col.dropna().shape[0] == 0:
            return pd.Series(np.nan, index=col.index)
        return (col - col.mean()) / col.std(ddof=0)

    df['StudentTeacherRatio_z'] = zscore(df['StudentTeacherRatio'])
    df['PctCalWorks_z'] = zscore(df['PctCalWorks'])
    df['PctReducedLunch_z'] = zscore(df['PctReducedLunch'])
    df['NumComputers_z'] = zscore(df['NumComputers'])
    df['ExpenditurePerStudent_z'] = zscore(df['ExpenditurePerStudent'])
    df['DistrictIncomeK_z'] = zscore(df['DistrictIncomeK'])
    df['PctEnglishLearners_z'] = zscore(df['PctEnglishLearners'])

    # Keep only columns needed for modeling plus a few useful originals for diagnostics
    keep_cols = [
        'TotalEnrollment', 'NumTeachers', 'StudentTeacherRatio', 'StudentTeacherRatio_z',
        'ReadingScore', 'MathScore', 'AvgTestScore',
        'PctCalWorks', 'PctCalWorks_z', 'PctReducedLunch', 'PctReducedLunch_z',
        'NumComputers', 'NumComputers_z', 'ExpenditurePerStudent', 'ExpenditurePerStudent_z',
        'DistrictIncomeK', 'DistrictIncomeK_z', 'PctEnglishLearners', 'PctEnglishLearners_z',
        'GradeSpan', 'GradeSpan_KK08', 'feature4'  # feature4 (County) preserved
    ]

    # Some datasets may not have all columns; intersect
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Run an OLS regression of AvgTestScore on student-teacher ratio (z-scored)
    and controls. Returns the fitted statsmodels RegressionResults object.

    Model specification:
      AvgTestScore_i = beta0 + beta1 * StudentTeacherRatio_z_i
                       + gamma' * Controls_z_i + delta * GradeSpan_KK08 + u_i

    Uses robust (HC3) standard errors.
    """
    # Ensure required columns exist (excluding AvgTestScore to avoid duplication)
    input_columns = ['StudentTeacherRatio_z',
                     'PctCalWorks_z', 'PctReducedLunch_z', 'NumComputers_z',
                     'ExpenditurePerStudent_z', 'DistrictIncomeK_z', 'PctEnglishLearners_z',
                     'GradeSpan_KK08']
    required = input_columns + ['AvgTestScore']

    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"The following required columns are missing from the transformed dataframe: {missing}")

    # Drop any remaining rows with NA in the model variables
    model_df = df[input_columns + ['AvgTestScore']].dropna()

    # Dependent variable (ensure it's a 1-d array / Series)
    y = model_df['AvgTestScore']

    # Design matrix
    X = model_df[input_columns]

    X = sm.add_constant(X)

    # Fit OLS with robust standard errors (HC3)
    ols_model = sm.OLS(y, X)
    results = ols_model.fit(cov_type='HC3')

    # Return the fitted results object so the caller can inspect summary, params, etc.
    return results