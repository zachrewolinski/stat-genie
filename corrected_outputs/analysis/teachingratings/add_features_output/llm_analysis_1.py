from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/teachingratings/add_features_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the Hamermesh classroom dataset for OLS modeling.

    Outputs a dataframe that contains the exact columns referenced in the conceptual model:
      - beauty, eval, age, gender, minority, tenure, native, division, credits,
        students, log_students, religiousness, prof

    Actions performed:
      - Keep only rows with non-missing values for the variables used in the model
      - Normalize/strip categorical string fields to lower-case and trim whitespace
      - Ensure students > 0 then create log_students = np.log(students)
      - Ensure numeric columns are properly typed
    """
    # Work on a copy
    df = df.copy()

    # Standardize string/categorical columns: strip whitespace and lower-case (where applicable)
    for col in ['minority', 'gender', 'credits', 'division', 'native', 'tenure']:
        if col in df.columns:
            # If column is non-string (e.g., categorical), convert to string then normalize
            df[col] = df[col].astype(str).str.strip().str.lower()
            # Replace known representations of missing values with actual NaN
            df.loc[df[col].isin(['nan', 'none', 'na', '']), col] = np.nan

    # Keep only columns required by the model (if present) and drop rows with missing values in these
    required_cols = ['beauty', 'eval', 'age', 'gender', 'minority', 'tenure', 'native',
                     'division', 'credits', 'students', 'religiousness', 'prof']

    # Intersect with columns that exist in the input df
    required_present = [c for c in required_cols if c in df.columns]

    # Drop rows missing any required_present columns
    df = df.dropna(subset=required_present)

    # Ensure numeric types for numeric fields
    numeric_cols = ['beauty', 'eval', 'age', 'students', 'religiousness', 'prof']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # After coercion, drop rows that became NA in any required numeric column
    df = df.dropna(subset=[c for c in numeric_cols if c in df.columns])

    # Remove rows with non-positive students (log undefined)
    if 'students' in df.columns:
        df = df[df['students'] > 0]
        # Create log_students (natural log)
        df['log_students'] = np.log(df['students'])

    # Keep the final set of columns used in modeling (ensures column names match conceptual variables)
    final_columns = []
    for c in ['beauty', 'eval', 'age', 'gender', 'minority', 'tenure', 'native',
              'division', 'credits', 'students', 'log_students', 'religiousness', 'prof']:
        if c in df.columns:
            final_columns.append(c)

    df = df[final_columns]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Estimate the effect of instructor beauty on teaching evaluations using OLS.

    Model specification (linear OLS with controls):
      eval ~ beauty + age + C(gender) + C(minority) + C(tenure) + C(native) + C(division) + C(credits) + log_students + religiousness

    Standard errors are clustered at the instructor (prof) level to account for multiple
    courses taught by the same instructor.

    Returns the fitted statsmodels results object (RegressionResultsWrapper).
    """
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['eval', 'beauty', 'age', 'gender', 'minority', 'tenure', 'native',
                'division', 'credits', 'log_students', 'religiousness', 'prof']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula
    formula = ('eval ~ beauty + age + C(gender) + C(minority) + C(tenure) + '
               'C(native) + C(division) + C(credits) + log_students + religiousness')

    # Fit OLS with instructor-level clustered standard errors
    ols_model = smf.ols(formula=formula, data=df)
    results = ols_model.fit(cov_type='cluster', cov_kwds={'groups': df['prof']})

    # Return the fitted results object (caller can call results.summary())
    return results


