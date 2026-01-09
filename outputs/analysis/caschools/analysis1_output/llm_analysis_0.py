from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/campus/austin.zane/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/caschools/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district-level dataframe to add the variables needed for modeling.

    Produces the following new columns used by the model:
      - StudentTeacherRatio: students / teachers
      - AvgScore: mean of read and math
      - ComputersPerStudent: computer / students

    Also coerces relevant columns to numeric, filters invalid rows (e.g., teachers <= 0),
    and drops rows missing any variable required for the regression.
    """

    # Make a copy to avoid modifying original
    df = df.copy()

    # Coerce key numeric columns to numeric (introduce NaN where parsing fails)
    numeric_cols = ['students', 'teachers', 'computer', 'expenditure', 'income', 'calworks', 'lunch', 'english', 'read', 'math']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows with invalid or zero teachers (can't compute ratio) or missing test scores
    if 'teachers' in df.columns:
        df.loc[df['teachers'] <= 0, 'teachers'] = np.nan

    # Compute dependent variable: average of reading and math scores
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Compute student-teacher ratio (students per teacher)
    df['StudentTeacherRatio'] = df['students'] / df['teachers']

    # Compute computers per student
    df['ComputersPerStudent'] = df['computer'] / df['students']

    # Keep columns needed for the model
    required_cols = [
        'StudentTeacherRatio', 'AvgScore', 'expenditure', 'income', 'calworks', 'lunch',
        'english', 'ComputersPerStudent', 'students', 'grades', 'county'
    ]

    # Some of these columns (grades, county) may be non-numeric — that's fine; we only ensure presence
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        # If required columns are missing from the input dataset, raise an informative error
        raise ValueError(f"Input dataframe is missing required columns for transformation: {missing_required}")

    # Drop rows with missing values in any of the required model columns
    df = df.dropna(subset=required_cols)

    # Optionally, convert grades and county to category dtype for modeling
    df['grades'] = df['grades'].astype('category')
    df['county'] = df['county'].astype('category')

    # Final sanity filter: keep only plausible StudentTeacherRatio values
    # (remove extreme/impossible ratios such as >1000 students per teacher)
    df = df[(df['StudentTeacherRatio'] > 0) & (df['StudentTeacherRatio'] < 1000)]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model of AvgScore on StudentTeacherRatio controlling for district resources
    and socioeconomic characteristics. Returns results with robust standard errors (HC3).

    Model formula:
      AvgScore ~ StudentTeacherRatio + expenditure + income + calworks + lunch + english
                 + ComputersPerStudent + students + C(grades) + C(county)

    Notes:
      - County and grades are included as categorical fixed effects using C(...).
      - Robust (heteroskedasticity-consistent) covariance is applied (HC3).
    """
    import statsmodels.formula.api as smf

    # Ensure required columns are present
    formula = (
        'AvgScore ~ StudentTeacherRatio + expenditure + income + calworks + lunch + english '
        '+ ComputersPerStudent + students + C(grades) + C(county)'
    )

    # Fit OLS
    ols_mod = smf.ols(formula=formula, data=df).fit()

    # Obtain robust covariance (HC3) for heteroskedasticity-robust inference
    results = ols_mod.get_robustcov_results(cov_type='HC3')

    # Print a concise summary for quick inspection
    print(results.summary())

    return results


