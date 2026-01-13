from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/replace_with_rvs_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district-level dataframe for modeling.

    Produces the following new columns used in the model:
      - StudentTeacherRatio: students / teachers (FTE)
      - AvgScore: mean of 'read' and 'math'
      - ComputerPerStudent: computer / students

    Keeps relevant control columns and drops rows with missing or invalid values in key columns.
    """
    df = df.copy()

    # Ensure numeric columns are present and coerce to numeric where appropriate
    numeric_cols = ['students', 'teachers', 'read', 'math', 'expenditure', 'income', 'english', 'lunch', 'computer', 'calworks']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the core outcome or exposure variables
    df = df.dropna(subset=['students', 'teachers', 'read', 'math'])

    # Remove rows with non-positive teachers or students to avoid invalid ratios
    df = df[df['teachers'] > 0]
    df = df[df['students'] > 0]

    # Construct Student-Teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = (df['read'] + df['math']) / 2.0

    # Computers per student
    df['ComputerPerStudent'] = df['computer'] / df['students']

    # Keep columns needed for modeling; include originals for reference
    keep_cols = [
        'StudentTeacherRatio', 'AvgScore', 'expenditure', 'income', 'english',
        'lunch', 'calworks', 'ComputerPerStudent', 'county', 'grades', 'students', 'teachers'
    ]

    # If any of the control columns are missing in the dataset, drop them from keep list gracefully
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df[keep_cols].copy()

    # Drop rows with any remaining missing values in the kept columns
    df = df.dropna()

    # Optional: reset index for downstream analysis
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model estimating the association between StudentTeacherRatio and AvgScore,
    controlling for district-level covariates and including categorical controls for county and grades.

    Returns the fitted results object (statsmodels RegressionResultsWrapper) with robust standard errors.
    """
    import statsmodels.formula.api as smf

    # Build formula. Use categorical controls for county and grades if present.
    formula_parts = [
        'StudentTeacherRatio',
        'expenditure',
        'income',
        'english',
        'lunch',
        'ComputerPerStudent',
        'calworks'
    ]

    # Only include the terms that exist in the dataframe
    formula_terms = [t for t in formula_parts if t in df.columns]

    # Add categorical variables if present
    if 'county' in df.columns:
        formula_terms.append('C(county)')
    if 'grades' in df.columns:
        formula_terms.append('C(grades)')

    formula = 'AvgScore ~ ' + ' + '.join(formula_terms)

    # Fit OLS with robust (HC3) standard errors
    model_fit = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    # Return the fitted model object for inspection (summary can be printed by the caller)
    return model_fit


