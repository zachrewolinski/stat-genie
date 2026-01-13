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
    Transform the raw district-level dataframe into the analysis-ready dataframe.

    Produces the following new columns used in modeling:
    - StudentTeacherRatio: students / teachers
    - AvgScore: mean of 'read' and 'math'
    - ComputersPerStudent: computer / students

    Also coerces numeric columns, drops rows with missing or invalid key values
    (e.g., teachers <= 0), and casts categorical controls.
    """
    df = df.copy()

    # Ensure key numeric columns are numeric (coerce errors to NaN)
    numeric_cols = [
        'students', 'teachers', 'expenditure', 'income', 'calworks',
        'lunch', 'computer', 'english', 'read', 'math'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the primary inputs required to compute the IV and DV
    required_for_ratio_and_scores = ['students', 'teachers', 'read', 'math']
    df = df.dropna(subset=[c for c in required_for_ratio_and_scores if c in df.columns])

    # Remove invalid teacher counts (avoid division by zero)
    if 'teachers' in df.columns:
        df = df[df['teachers'] > 0]

    # Compute Student-Teacher Ratio
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Compute average academic score (dependent variable)
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Compute computers per student as a technology access control
    # protect against division by zero; students should be > 0 by prior drop
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Ensure categorical controls are categorical
    if 'grades' in df.columns:
        df['grades'] = df['grades'].astype('category')
    if 'county' in df.columns:
        df['county'] = df['county'].astype('category')

    # Drop rows with missing values in key control variables to keep model listwise
    controls_to_check = [c for c in ['expenditure', 'income', 'calworks', 'lunch', 'english', 'ComputersPerStudent'] if c in df.columns]
    if controls_to_check:
        df = df.dropna(subset=controls_to_check)

    # Final check: make sure the columns required by the model exist
    required_model_cols = ['StudentTeacherRatio', 'AvgScore', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'ComputersPerStudent', 'grades', 'county']
    # Keep only rows that have these columns present (if some controls were not in original data they will be skipped by model construction)
    # Return the transformed dataframe
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression of average academic performance on student-teacher ratio
    controlling for district resources and demographics. Uses robust (HC3) standard errors.

    Model formula:
    AvgScore ~ StudentTeacherRatio + expenditure + income + calworks + lunch + english + ComputersPerStudent + C(grades) + C(county)

    Returns the fitted statsmodels regression results object.
    """
    import statsmodels.formula.api as smf

    # Ensure dataframe passed in is the transformed dataframe
    df = df.copy()

    # Build formula. If 'county' or 'grades' are not present in df, remove those terms safely.
    formula_parts = [
        'StudentTeacherRatio',
        'expenditure',
        'income',
        'calworks',
        'lunch',
        'english',
        'ComputersPerStudent'
    ]

    # Add categorical terms only if present
    if 'grades' in df.columns:
        formula_parts.append('C(grades)')
    if 'county' in df.columns:
        formula_parts.append('C(county)')

    formula = 'AvgScore ~ ' + ' + '.join(formula_parts)

    # Fit OLS with robust standard errors (HC3)
    model = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    # Return the fitted model object (caller can examine summary, params, etc.)
    return model


