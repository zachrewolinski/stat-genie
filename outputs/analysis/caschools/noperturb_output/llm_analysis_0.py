from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/noperturb_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district dataframe into a modeling dataframe.

    Adds/returns the following columns (all required for the model):
      - AvgScore: mean of 'read' and 'math'
      - StudentTeacherRatio: students / teachers
      - StudentTeacherRatio_log: ln(StudentTeacherRatio)
      - Expenditure: from 'expenditure'
      - PercentFreeLunch: from 'lunch'
      - PercentEnglishLearners: from 'english'
      - PercentCalWorks: from 'calworks'
      - Income: from 'income' (district average income, in thousands)
      - ComputersPerStudent: computer / students
      - Grades_KK08: indicator if grades == 'KK-08'
      - County: string version of 'county' (for fixed effects)

    Drops rows with missing or invalid values in the required columns.
    """
    df = df.copy()

    # Ensure required original columns exist
    required_orig = ['students', 'teachers', 'read', 'math', 'expenditure', 'lunch', 'english', 'calworks', 'income', 'computer', 'grades', 'county']
    missing_cols = [c for c in required_orig if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Input dataframe is missing required columns: {missing_cols}")

    # Drop rows missing the critical numeric inputs
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove or mark non-positive teacher counts to avoid division by zero
    # (set to NaN so they will be dropped below)
    df.loc[df['teachers'] <= 0, 'teachers'] = np.nan
    df = df.dropna(subset=['teachers'])

    # Compute student-teacher ratio and log transform
    df['StudentTeacherRatio'] = df['students'] / df['teachers']
    # If ratio is non-positive or NaN, set to NaN
    df.loc[~np.isfinite(df['StudentTeacherRatio']) | (df['StudentTeacherRatio'] <= 0), 'StudentTeacherRatio'] = np.nan
    df['StudentTeacherRatio_log'] = np.log(df['StudentTeacherRatio'])

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Controls: carry forward and rename to clear modeling column names
    df['Expenditure'] = df['expenditure']
    df['PercentFreeLunch'] = df['lunch']
    df['PercentEnglishLearners'] = df['english']
    df['PercentCalWorks'] = df['calworks']
    df['Income'] = df['income']

    # Computers per student (handle zero/NA students)
    df['ComputersPerStudent'] = df['computer'] / df['students']
    df.loc[~np.isfinite(df['ComputersPerStudent']), 'ComputersPerStudent'] = np.nan

    # Grades indicator
    df['Grades_KK08'] = (df['grades'].astype(str) == 'KK-08').astype(int)

    # County as string for fixed effects
    df['County'] = df['county'].astype(str)

    # Final model columns required
    model_cols = [
        'AvgScore',
        'StudentTeacherRatio',
        'StudentTeacherRatio_log',
        'Expenditure',
        'PercentFreeLunch',
        'PercentEnglishLearners',
        'PercentCalWorks',
        'Income',
        'ComputersPerStudent',
        'Grades_KK08',
        'County'
    ]

    # Drop rows missing any of the model columns
    df = df.dropna(subset=model_cols)

    # Return dataframe that contains at least the model columns
    # (we keep only model columns to make the modeling dataframe compact)
    return df[model_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression of AvgScore on StudentTeacherRatio (log) and controls.

    Model specification:
      AvgScore ~ StudentTeacherRatio_log + Expenditure + PercentFreeLunch
                 + PercentEnglishLearners + PercentCalWorks + Income
                 + ComputersPerStudent + Grades_KK08 + C(County)

    Uses robust (HC3) standard errors. Also computes VIFs for numeric predictors
    (excluding County fixed effects) to help diagnose multicollinearity.

    Returns a dict with keys:
      - 'fitted_model': the statsmodels regression results object
      - 'summary_text': text summary (res.summary().as_text())
      - 'vif': pandas DataFrame of VIF values
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Ensure required model columns exist
    required_model_cols = ['AvgScore', 'StudentTeacherRatio_log', 'Expenditure', 'PercentFreeLunch',
                           'PercentEnglishLearners', 'PercentCalWorks', 'Income', 'ComputersPerStudent',
                           'Grades_KK08', 'County']
    missing = [c for c in required_model_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Transformed dataframe missing required columns: {missing}")

    # Build formula with county fixed effects
    formula = (
        'AvgScore ~ StudentTeacherRatio_log + Expenditure + PercentFreeLunch '
        '+ PercentEnglishLearners + PercentCalWorks + Income '
        '+ ComputersPerStudent + Grades_KK08 + C(County)'
    )

    # Fit OLS with robust standard errors (HC3)
    fitted = smf.ols(formula, data=df).fit(cov_type='HC3')

    # Compute VIF for numeric predictors (exclude County categorical variable)
    vif_vars = ['StudentTeacherRatio_log', 'Expenditure', 'PercentFreeLunch', 'PercentEnglishLearners',
                'PercentCalWorks', 'Income', 'ComputersPerStudent', 'Grades_KK08']
    X = df[vif_vars].copy()
    X = sm.add_constant(X)
    vif_values = []
    for i in range(X.shape[1]):
        try:
            vif_values.append(variance_inflation_factor(X.values, i))
        except Exception:
            vif_values.append(np.nan)
    vif = pd.DataFrame({'variable': X.columns, 'VIF': vif_values})

    return {
        'fitted_model': fitted,
        'summary_text': fitted.summary().as_text(),
        'vif': vif
    }