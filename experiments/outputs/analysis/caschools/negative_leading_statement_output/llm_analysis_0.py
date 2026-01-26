from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/negative_leading_statement_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district-level dataframe to produce the final modeling dataframe.

    Produces the following new columns used by the model:
      - AvgScore: mean of 'read' and 'math'
      - StudentTeacherRatio: students / teachers (winsorized at 1st/99th percentiles)
      - ComputersPerStudent: computer / students
      - LogStudents: natural log of students

    Drops rows with missing or invalid values required for modeling.
    """
    df = df.copy()

    # Ensure numeric columns are numeric
    numeric_cols = ['students', 'teachers', 'read', 'math', 'computer', 'expenditure', 'income', 'calworks', 'lunch', 'english']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the essential outcome or denominator for ratio
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove invalid teacher counts (must be > 0 for ratio)
    df = df[df['teachers'] > 0]

    # Dependent variable: average of reading and math
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Independent variable: student-teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Winsorize the ratio to reduce influence of extreme outliers (1st/99th percentiles)
    if not df['StudentTeacherRatio'].isna().all():
        low = df['StudentTeacherRatio'].quantile(0.01)
        high = df['StudentTeacherRatio'].quantile(0.99)
        df['StudentTeacherRatio'] = df['StudentTeacherRatio'].clip(lower=low, upper=high)

    # Computer resources normalized by student body size
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Log students to capture scale nonlinearity
    # Guard against nonpositive students (already filtered), but be safe
    df['LogStudents'] = np.log(df['students'].replace(0, np.nan))

    # Ensure categorical controls exist and have no missing values for modeling
    # We'll keep 'grades' and 'county' as categorical variables in the model (C(...))
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype('category')
    if 'county' in df.columns:
        df['county'] = df['county'].astype('category')

    # Drop rows with missing values in controls used in the model
    control_cols = ['expenditure', 'income', 'calworks', 'lunch', 'english', 'ComputersPerStudent', 'LogStudents', 'grades', 'county']
    # Some of these (grades, county) may be non-numeric; dropna will handle them
    df = df.dropna(subset=[c for c in control_cols if c in df.columns])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Runs an OLS regression testing the association between student-teacher ratio and average test scores,
    controlling for district resources and demographics. Returns the fitted statsmodels results instance.

    Model specification:
      AvgScore ~ StudentTeacherRatio + I(StudentTeacherRatio**2) + expenditure + income + calworks + lunch + english
                 + ComputersPerStudent + LogStudents + C(grades) + C(county)

    The quadratic term allows for nonlinearity (diminishing returns). County and grades are included as categorical
    fixed effects using C(...). Robust (HC3) standard errors are requested to mitigate heteroskedasticity.
    """
    import statsmodels.formula.api as smf

    # Copy to avoid mutating caller's dataframe
    data = df.copy()

    # Formula including a quadratic term for possible nonlinearity and categorical fixed effects
    formula = (
        'AvgScore ~ StudentTeacherRatio + I(StudentTeacherRatio**2) '
        '+ expenditure + income + calworks + lunch + english '
        '+ ComputersPerStudent + LogStudents + C(grades) + C(county)'
    )

    # Fit OLS with robust standard errors (HC3)
    model_res = smf.ols(formula, data=data).fit(cov_type='HC3')

    # Print a concise summary for quick inspection (caller can use the returned object for detailed analysis)
    print(model_res.summary())

    # Return the fitted model results object for downstream inspection (coefficients, p-values, CIs, residuals, etc.)
    return model_res


