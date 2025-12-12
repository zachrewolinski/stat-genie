from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/caschools/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw district-level dataframe to produce the variables required for modeling.

    Produces:
    - ST_Ratio: students / teachers
    - log_ST_Ratio: natural log of ST_Ratio (used in the model)
    - AvgScore: mean of 'read' and 'math'
    - ComputerPerStudent: computer / students
    - standardized (z) versions of continuous controls used in the regression
    - drops rows with missing/invalid values for key variables
    """
    df = df.copy()

    # Ensure key numeric columns are numeric (coerce invalid values to NaN)
    numeric_cols = ['students', 'teachers', 'read', 'math', 'computer', 'expenditure', 'lunch', 'english', 'income']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing essential columns for computing ST_Ratio and outcome
    required_for_computation = ['students', 'teachers', 'read', 'math']
    df = df.dropna(subset=[c for c in required_for_computation if c in df.columns])

    # Remove rows where teachers is zero or negative (invalid for ratio)
    df = df[df['teachers'] > 0]

    # Student-teacher ratio and log transform
    df['ST_Ratio'] = df['students'] / df['teachers']

    # If ST_Ratio is non-positive for any reason, set to NA (should already be removed by teachers>0)
    df.loc[df['ST_Ratio'] <= 0, 'ST_Ratio'] = np.nan
    df['log_ST_Ratio'] = np.log(df['ST_Ratio'])

    # Dependent variable: average of reading and math scores
    df['AvgScore'] = (df['read'] + df['math']) / 2.0

    # Resource measure: computers per student
    df['ComputerPerStudent'] = np.where(df['students'] > 0, df['computer'] / df['students'], np.nan)

    # Create standardized (z-score) versions of continuous controls used in the model
    # Use population std (ddof=0) for consistency
    z_columns = ['expenditure', 'lunch', 'english', 'income', 'ComputerPerStudent']
    for col in z_columns:
        if col in df.columns:
            col_vals = df[col]
            mean = col_vals.mean(skipna=True)
            std = col_vals.std(ddof=0, skipna=True)
            # If std is zero or NaN, create a column of NaNs to avoid dividing by zero
            if pd.isna(std) or std == 0:
                df[col + '_z'] = np.nan
            else:
                df[col + '_z'] = (col_vals - mean) / std
        else:
            df[col + '_z'] = np.nan

    # Keep only rows that have the variables used in the regression model
    model_required = ['log_ST_Ratio', 'AvgScore', 'expenditure_z', 'lunch_z', 'english_z', 'income_z', 'ComputerPerStudent_z', 'grades', 'county']
    # Drop rows missing any of these required columns
    existing_required = [c for c in model_required if c in df.columns]
    df = df.dropna(subset=existing_required)

    # Return final dataframe (includes all created columns). Caller can further subset if needed.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Estimate the relationship between student-teacher ratio and academic performance.

    Model specification (OLS with robust standard errors):
      AvgScore ~ log_ST_Ratio + expenditure_z + lunch_z + english_z + income_z + ComputerPerStudent_z + C(grades) + C(county)

    Returns:
    - statsmodels regression results object (fitted model)
    """
    import statsmodels.formula.api as smf

    # Ensure the dataframe passed in contains the columns created by transform
    required = ['AvgScore', 'log_ST_Ratio', 'expenditure_z', 'lunch_z', 'english_z', 'income_z', 'ComputerPerStudent_z', 'grades', 'county']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"The following required columns are missing from the dataframe: {missing}")

    # Define formula: county and grades included as categorical factors
    formula = (
        'AvgScore ~ log_ST_Ratio + expenditure_z + lunch_z + english_z + '
        'income_z + ComputerPerStudent_z + C(grades) + C(county)'
    )

    # Fit OLS with heteroskedasticity-robust standard errors (HC3)
    results = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    # Print summary for convenience and return the results object for downstream inspection
    try:
        print(results.summary())
    except Exception:
        # If printing fails in some environments, ignore and still return results
        pass

    return results


