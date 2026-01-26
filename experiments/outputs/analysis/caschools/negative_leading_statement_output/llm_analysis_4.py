from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district-level dataframe into the analysis-ready dataframe.

    Produces the following required columns (exact names used in the model):
      - StudentTeacherRatio: students / teachers
      - AvgScore: (Read + Math) / 2
      - Read, Math: preserved
      - Expenditure, Income, CalWorks, Lunch, English: cleaned numeric controls
      - ComputersPerStudent: computer / students
      - LogStudents: np.log(students)
      - IsKK08: indicator for grades == 'KK-08'
      - county: preserved categorical variable for potential fixed effects

    The function drops rows with missing essential variables (students, teachers, read, math).
    """

    # Copy to avoid modifying original
    df = df.copy()

    # Standardize column names if necessary (assume input columns are as provided in schema)
    # Drop rows missing the essential variables for computing the key measures
    essential_cols = ['students', 'teachers', 'read', 'math']
    df = df.dropna(subset=essential_cols)

    # Compute student-teacher ratio
    # Protect against zero teachers (shouldn't happen in valid data); drop such rows
    df = df[df['teachers'] > 0]
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Dependent variable: average of reading and math scores
    df['Read'] = df['read']
    df['Math'] = df['math']
    df['AvgScore'] = (df['Read'] + df['Math']) / 2.0

    # Controls: bring forward and sanitize names (match exact column names used in model)
    df['Expenditure'] = df['expenditure']
    df['Income'] = df['income']
    df['CalWorks'] = df['calworks']
    df['Lunch'] = df['lunch']
    df['English'] = df['english']

    # Computers per student: if computer or students missing, those rows have been dropped above
    # Protect division by zero: students > 0 ensured by earlier drop
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Log students to capture scale nonlinearity
    df['LogStudents'] = np.log(df['students'].astype(float))

    # Grade-span indicator: 1 if KK-08 (K-8), 0 otherwise. Preserve exact string matching.
    df['IsKK08'] = df['grades'].astype(str).apply(lambda x: 1 if x.strip() == 'KK-08' else 0)

    # Preserve county column (categorical) for potential fixed effects
    df['county'] = df['county'].astype(str)

    # If any of the numeric control columns are missing, drop those rows (keeps sample consistent)
    control_cols = ['Expenditure', 'Income', 'CalWorks', 'Lunch', 'English', 'ComputersPerStudent']
    df = df.dropna(subset=control_cols)

    # Optional: Winsorize or clip StudentTeacherRatio to remove extreme outliers that could unduly influence OLS
    # Here we cap ratios at the 1st and 99th percentiles
    lower = df['StudentTeacherRatio'].quantile(0.01)
    upper = df['StudentTeacherRatio'].quantile(0.99)
    df['StudentTeacherRatio'] = df['StudentTeacherRatio'].clip(lower, upper)

    # Return only columns needed for modeling (keeps the dataframe compact)
    keep_cols = [
        'StudentTeacherRatio', 'AvgScore', 'Read', 'Math',
        'Expenditure', 'Income', 'CalWorks', 'Lunch', 'English',
        'ComputersPerStudent', 'LogStudents', 'IsKK08', 'county'
    ]
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """\n    Fit OLS regressions to test whether lower student-teacher ratio is associated with higher academic performance.\n\n    Primary specification:\n      AvgScore ~ StudentTeacherRatio + controls\n    Controls included: Expenditure, Income, CalWorks, Lunch, English, ComputersPerStudent, LogStudents, IsKK08.\n    County fixed effects (dummies) are added to account for regional differences.\n\n    Robust (HC3) standard errors are used. Returns a dictionary with fitted model objects for\n    average score (primary) and separate read/math specifications (sensitivity).\n    """
    # Ensure the expected columns are present
    required = ['StudentTeacherRatio', 'AvgScore', 'Read', 'Math',
                'Expenditure', 'Income', 'CalWorks', 'Lunch', 'English',
                'ComputersPerStudent', 'LogStudents', 'IsKK08', 'county']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Prepare design matrix
    control_vars = [
        'Expenditure', 'Income', 'CalWorks', 'Lunch', 'English',
        'ComputersPerStudent', 'LogStudents', 'IsKK08'
    ]

    # Create county dummies for fixed effects (drop one to avoid multicollinearity)
    county_dummies = pd.get_dummies(df['county'], prefix='county', drop_first=True)

    X_base = df[control_vars].astype(float)
    X = pd.concat([df[['StudentTeacherRatio']].astype(float), X_base, county_dummies], axis=1)
    X = sm.add_constant(X)

    # Primary model: AvgScore
    y_avg = df['AvgScore'].astype(float)
    model_avg = sm.OLS(y_avg, X).fit(cov_type='HC3')

    # Sensitivity: separate models for Read and Math
    y_read = df['Read'].astype(float)
    model_read = sm.OLS(y_read, X).fit(cov_type='HC3')

    y_math = df['Math'].astype(float)
    model_math = sm.OLS(y_math, X).fit(cov_type='HC3')

    # Additional robustness: include quadratic term for StudentTeacherRatio to test nonlinearity
    X_quad = X.copy()
    X_quad['StudentTeacherRatio_sq'] = X_quad['StudentTeacherRatio'] ** 2
    model_avg_quad = sm.OLS(y_avg, X_quad).fit(cov_type='HC3')

    # Optional diagnostics: compute VIFs for non-county predictors
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        # compute VIF on the non-constant, non-county columns
        vif_vars = ['StudentTeacherRatio'] + control_vars
        vif_X = sm.add_constant(df[vif_vars].astype(float))
        vifs = {var: variance_inflation_factor(vif_X.values, i + 1) for i, var in enumerate(vif_vars)}
    except Exception:
        vifs = None

    # Return models and diagnostics
    results = {
        'model_avg': model_avg,
        'model_read': model_read,
        'model_math': model_math,
        'model_avg_quad': model_avg_quad,
        'vifs': vifs
    }
    return results


if __name__ == "__main__":
    # Example usage guarded under main to avoid execution on import
    try:
        df_raw = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/negative_leading_statement_output/caschools.csv')
        df_final = transform(df_raw)
        res = model(df_final)
        print("Models fitted successfully. Primary model summary:")
        print(res['model_avg'].summary())
    except Exception as e:
        print(f"Example run failed: {e}")