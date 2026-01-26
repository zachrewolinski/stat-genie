from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/caschools/add_features_output/caschools.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original dataframe to produce variables used in the statistical model.

    Outputs (added columns):
      - StudentTeacherRatio: students / teachers
      - AvgScore: mean of read and math
      - ComputersPerStudent: computer / students
      - grades_KK08: binary indicator for grades == 'KK-08'
      - standardized versions (z-scores) for DV, IV, and continuous controls:
          AvgScore_z, STR_z, expenditure_z, lunch_z, english_z, calworks_z,
          ComputersPerStudent_z, income_z, students_z

    The function drops rows with missing or invalid critical values (students, teachers, read, math).
    """

    # Make a working copy
    df = df.copy()

    # Ensure numeric columns are numeric
    num_cols = ['students', 'teachers', 'read', 'math', 'computer', 'expenditure', 'lunch', 'english', 'calworks', 'income']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing essential fields for computing ratio and outcome
    required = [c for c in ['students', 'teachers', 'read', 'math'] if c in df.columns]
    df = df.dropna(subset=required)

    # Remove invalid or zero teachers to avoid division by zero
    df = df[df['teachers'] > 0]
    df = df[df['students'] > 0]

    # Compute student-teacher ratio and average score
    df['StudentTeacherRatio'] = df['students'] / df['teachers']
    df['AvgScore'] = df[['read', 'math']].mean(axis=1)

    # Computers per student (if computer column exists)
    if 'computer' in df.columns:
        # Avoid dividing by zero; we already filtered students > 0
        df['ComputersPerStudent'] = df['computer'] / df['students']
    else:
        df['ComputersPerStudent'] = np.nan

    # Binary indicator for grade span KK-08 vs KK-06
    # Create a column grades_KK08 = 1 if grades == 'KK-08', else 0 (including missing treated as 0)
    if 'grades' in df.columns:
        df['grades_KK08'] = df['grades'].astype(str).apply(lambda x: 1 if x == 'KK-08' else 0)
    else:
        df['grades_KK08'] = 0

    # Keep county for clustering (preserve original values)
    if 'county' not in df.columns:
        df['county'] = np.nan

    # Select continuous controls to standardize (z-score)
    to_standardize = {
        'STR': 'StudentTeacherRatio',
        'AvgScore': 'AvgScore',
        'expenditure': 'expenditure',
        'lunch': 'lunch',
        'english': 'english',
        'calworks': 'calworks',
        'ComputersPerStudent': 'ComputersPerStudent',
        'income': 'income',
        'students': 'students'
    }

    # Compute z-scores and place in columns with suffix _z
    for short, col in to_standardize.items():
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std(ddof=0)
            # If std is zero or NaN, fill result with zeros to avoid division by zero
            if pd.isna(std) or std == 0:
                df[f"{short}_z"] = 0.0
            else:
                df[f"{short}_z"] = (df[col] - mean) / std
        else:
            df[f"{short}_z"] = np.nan

    # For clarity, rename the StudentTeacherRatio z column to match the conceptual name used in the model
    # The mapping above created 'STR_z' and 'AvgScore_z' etc.

    # Keep only rows with a non-missing AvgScore_z and STR_z (model requires both)
    df = df.dropna(subset=['AvgScore_z', 'STR_z'])

    # Final dataframe returned contains the original columns plus the derived ones used in modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear regression of standardized average score on standardized student-teacher ratio
    controlling for relevant district-level covariates. Cluster-robust standard errors by county.

    Model formula:
      AvgScore_z ~ STR_z + expenditure_z + lunch_z + english_z + calworks_z
                 + ComputersPerStudent_z + income_z + students_z + grades_KK08

    Returns the fitted statsmodels results object (OLSResults).
    """

    import statsmodels.formula.api as smf

    # Ensure required columns are available
    required_cols = ['AvgScore_z', 'STR_z', 'expenditure_z', 'lunch_z', 'english_z', 'calworks_z',
                     'ComputersPerStudent_z', 'income_z', 'students_z', 'grades_KK08', 'county']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"The following required columns are missing from the dataframe passed to model(): {missing}")

    # Drop rows with missing values in the predictors or outcome
    model_df = df.dropna(subset=['AvgScore_z', 'STR_z'])

    # Define formula
    formula = (
        'AvgScore_z ~ STR_z + expenditure_z + lunch_z + english_z + calworks_z '
        '+ ComputersPerStudent_z + income_z + students_z + grades_KK08'
    )

    # Fit OLS
    ols_mod = smf.ols(formula=formula, data=model_df)

    # Use cluster-robust standard errors clustered by county if county has enough groups
    # fall back to default if county is all NA or single group
    try:
        # If county has only one unique non-null value, clustering will fail; check that
        valid_counties = model_df['county'].dropna().unique()
        if len(valid_counties) > 1:
            results = ols_mod.fit(cov_type='cluster', cov_kwds={'groups': model_df['county']})
        else:
            results = ols_mod.fit()
    except Exception:
        # If clustering raises an error for any reason, use default OLS fit
        results = ols_mod.fit()

    # Return the fitted results object so the caller can inspect params, summary, etc.
    return results


