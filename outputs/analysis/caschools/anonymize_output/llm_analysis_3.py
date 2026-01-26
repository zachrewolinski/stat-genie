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
    Transform the raw dataset (feature1..feature15) into a dataframe with the exact columns
    used by the model. The function creates:
      - AvgScore: mean of feature14 (reading) and feature15 (math)
      - Enrollment, NumTeachers, StudentTeacherRatio, LogStudentTeacherRatio
      - PercentReducedLunch, PercentEngLearners, ExpenditurePerStudent,
        PercentCalWorks, DistrictIncome_k, NumComputers
      - County (categorical) and GradeSpan (categorical)

    It also drops rows with missing key variables and winsorizes StudentTeacherRatio at 1st/99th percentiles.
    """
    df = df.copy()

    # Map original columns to descriptive names used in modeling
    # feature6: Total enrollment
    # feature7: Number of teachers
    # feature8: Percent qualifying for CalWorks
    # feature9: Percent qualifying for reduced-price lunch
    # feature10: Number of computers
    # feature11: Expenditure per student
    # feature12: District average income (in 1,000 USD)
    # feature13: Percent English learners
    # feature14: Average reading score
    # feature15: Average math score
    # feature4: County
    # feature5: Grade span

    df['Enrollment'] = df['feature6']
    df['NumTeachers'] = df['feature7']
    df['PercentCalWorks'] = df['feature8']
    df['PercentReducedLunch'] = df['feature9']
    df['NumComputers'] = df['feature10']
    df['ExpenditurePerStudent'] = df['feature11']
    df['DistrictIncome_k'] = df['feature12']
    df['PercentEngLearners'] = df['feature13']
    df['AvgReading'] = df['feature14']
    df['AvgMath'] = df['feature15']
    df['County'] = df['feature4']
    df['GradeSpan'] = df['feature5']

    # Create dependent variable: average of reading and math
    df['AvgScore'] = df[['AvgReading', 'AvgMath']].mean(axis=1)

    # Create student-teacher ratio; guard against division by zero
    df['NumTeachers'] = pd.to_numeric(df['NumTeachers'], errors='coerce')
    df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce')
    df.loc[df['NumTeachers'] == 0, 'NumTeachers'] = np.nan
    df['StudentTeacherRatio'] = df['Enrollment'] / df['NumTeachers']

    # Log transform of ratio for alternative specification
    df['LogStudentTeacherRatio'] = np.log(df['StudentTeacherRatio'].replace(0, np.nan))

    # Winsorize StudentTeacherRatio to reduce influence of extreme outliers (1st-99th percentile)
    valid_ratio = df['StudentTeacherRatio'].dropna()
    if len(valid_ratio) > 0:
        lower = np.percentile(valid_ratio, 1)
        upper = np.percentile(valid_ratio, 99)
        df['StudentTeacherRatio'] = df['StudentTeacherRatio'].clip(lower=lower, upper=upper)
        # recompute log after clipping
        df['LogStudentTeacherRatio'] = np.log(df['StudentTeacherRatio'].replace(0, np.nan))

    # Ensure numeric controls are numeric
    numeric_cols = [
        'PercentCalWorks', 'PercentReducedLunch', 'NumComputers', 'ExpenditurePerStudent',
        'DistrictIncome_k', 'PercentEngLearners'
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the outcome or main IV or essential controls
    required = ['AvgScore', 'StudentTeacherRatio', 'PercentReducedLunch', 'PercentEngLearners',
                'ExpenditurePerStudent']
    df = df.dropna(subset=required)

    # Keep only the columns needed for modeling (plus a few descriptive columns)
    keep_cols = [
        'AvgScore', 'StudentTeacherRatio', 'LogStudentTeacherRatio', 'PercentReducedLunch',
        'PercentEngLearners', 'ExpenditurePerStudent', 'PercentCalWorks', 'DistrictIncome_k',
        'Enrollment', 'NumTeachers', 'NumComputers', 'County', 'GradeSpan'
    ]
    # Some of these may not exist if original data lacked them; filter to existing
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit OLS regressions to estimate the association between student-teacher ratio and average test scores.
    Returns two specifications with robust (HC3) standard errors:
      - model_linear: AvgScore ~ StudentTeacherRatio + controls + county & grade-span dummies
      - model_log:    AvgScore ~ LogStudentTeacherRatio + controls + county & grade-span dummies

    The function returns a dictionary with the fitted results objects.
    """
    results = {}
    data = df.copy()

    # Prepare categorical dummies for county and grade span (if present)
    cat_cols = []
    if 'County' in data.columns:
        data['County'] = data['County'].astype('category')
        county_dummies = pd.get_dummies(data['County'], prefix='County', drop_first=True)
        cat_cols.append('County')
    else:
        county_dummies = pd.DataFrame(index=data.index)

    if 'GradeSpan' in data.columns:
        data['GradeSpan'] = data['GradeSpan'].astype('category')
        grade_dummies = pd.get_dummies(data['GradeSpan'], prefix='GradeSpan', drop_first=True)
        cat_cols.append('GradeSpan')
    else:
        grade_dummies = pd.DataFrame(index=data.index)

    # Base set of controls
    control_vars = [
        'PercentReducedLunch', 'PercentEngLearners', 'ExpenditurePerStudent',
        'PercentCalWorks', 'DistrictIncome_k', 'Enrollment', 'NumComputers'
    ]
    control_vars = [c for c in control_vars if c in data.columns]

    # Build design matrices for the two specifications
    # Specification 1: linear ratio
    X_lin = pd.DataFrame(index=data.index)
    X_lin['StudentTeacherRatio'] = data['StudentTeacherRatio']
    for c in control_vars:
        X_lin[c] = data[c]
    # append categorical dummies
    X_lin = pd.concat([X_lin, county_dummies, grade_dummies], axis=1)
    X_lin = sm.add_constant(X_lin, has_constant='add')

    # Specification 2: log ratio
    X_log = pd.DataFrame(index=data.index)
    if 'LogStudentTeacherRatio' in data.columns:
        X_log['LogStudentTeacherRatio'] = data['LogStudentTeacherRatio']
    else:
        # fall back to log of ratio if available
        X_log['LogStudentTeacherRatio'] = np.log(data['StudentTeacherRatio'].replace(0, np.nan))

    for c in control_vars:
        X_log[c] = data[c]
    X_log = pd.concat([X_log, county_dummies, grade_dummies], axis=1)
    X_log = sm.add_constant(X_log, has_constant='add')

    y = data['AvgScore']

    # Drop any rows with NA in X or y for each specification
    X1 = X_lin.dropna()
    y1 = y.loc[X1.index]
    model_linear = sm.OLS(y1, X1).fit(cov_type='HC3')
    results['model_linear'] = model_linear

    X2 = X_log.dropna()
    y2 = y.loc[X2.index]
    model_log = sm.OLS(y2, X2).fit(cov_type='HC3')
    results['model_log'] = model_log

    # Also include a quick summary table of key coefficients
    summary_table = pd.DataFrame({
        'coef_linear': model_linear.params.reindex(['StudentTeacherRatio'], fill_value=np.nan),
        'se_linear': model_linear.bse.reindex(['StudentTeacherRatio'], fill_value=np.nan),
        'coef_log': model_log.params.reindex(['LogStudentTeacherRatio'], fill_value=np.nan),
        'se_log': model_log.bse.reindex(['LogStudentTeacherRatio'], fill_value=np.nan)
    })
    results['summary_table'] = summary_table

    return results


