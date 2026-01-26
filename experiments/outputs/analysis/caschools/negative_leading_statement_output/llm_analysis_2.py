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
    Transform the input dataframe into the analysis-ready dataframe.

    Produces the following new columns used in modeling:
      - StudentTeacherRatio: students / teachers (must have teachers > 0)
      - AcademicScore: mean of read and math scores
      - ComputersPerStudent: computer / students
      - LogStudents: natural log of students

    Also imputes medians for a small set of numeric controls to avoid excessive row loss,
    and drops rows with missing essential fields (students, teachers, read, math).
    """
    df = df.copy()

    # Essential columns that must be present for the analysis
    essential = ['students', 'teachers', 'read', 'math']
    # Drop rows missing essential columns
    df = df.dropna(subset=essential)

    # Remove invalid teacher counts (zero or negative) to avoid division by zero
    df = df[df['teachers'] > 0]

    # Compute Student-Teacher Ratio
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Dependent variable: average of read and math
    df['AcademicScore'] = df[['read', 'math']].mean(axis=1)

    # Resource / technology control: computers per student
    # Avoid division by zero (students > 0 ensured by earlier drop)
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Log enrollment to capture non-linear size effects
    # Add a small constant for numerical stability (not strictly necessary since students>0)
    df['LogStudents'] = np.log(df['students'].astype(float))

    # Columns to impute with median if missing (to retain records while controlling)
    median_impute_cols = ['income', 'calworks', 'lunch', 'english', 'expenditure', 'computer']
    for col in median_impute_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Ensure categorical controls are treated as such
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype('category')
    if 'county' in df.columns:
        # Convert county to string category to use as fixed effect in modeling
        df['county'] = df['county'].astype('category')

    # Final drop: ensure no missing values in modeling columns
    model_cols = [
        'AcademicScore', 'StudentTeacherRatio', 'income', 'calworks', 'lunch', 'english',
        'expenditure', 'ComputersPerStudent', 'LogStudents', 'grades', 'county'
    ]
    # Keep only columns that actually exist in df (some datasets might lack some controls)
    model_cols = [c for c in model_cols if c in df.columns]
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit an OLS model testing whether lower student-teacher ratio is associated with higher
    academic performance, controlling for socioeconomic and resource variables.

    Model specification (linear + quadratic for ratio to allow nonlinearity):
      AcademicScore ~ StudentTeacherRatio + StudentTeacherRatio^2 + income + calworks + lunch
                      + english + expenditure + ComputersPerStudent + LogStudents
                      + categorical(grades) + categorical(county)

    Returns the fitted statsmodels result object (with HC3 robust standard errors).
    """
    import statsmodels.formula.api as smf

    # Check that necessary columns are present
    required = ['AcademicScore', 'StudentTeacherRatio']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")

    # Build formula dynamically depending on which controls are present
    controls = []
    candidate_controls = ['income', 'calworks', 'lunch', 'english', 'expenditure',
                          'ComputersPerStudent', 'LogStudents']
    for c in candidate_controls:
        if c in df.columns:
            controls.append(c)

    # Include categorical controls if present
    cat_controls = []
    if 'grades' in df.columns:
        cat_controls.append('C(grades)')
    if 'county' in df.columns:
        cat_controls.append('C(county)')

    rhs_terms = ['StudentTeacherRatio', 'I(StudentTeacherRatio**2)'] + controls + cat_controls
    formula = 'AcademicScore ~ ' + ' + '.join(rhs_terms)

    # Fit OLS with robust (HC3) standard errors
    model_fit = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    # Return the fitted model object (user can call .summary() or inspect params)
    return model_fit


