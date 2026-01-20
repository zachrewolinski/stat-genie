from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/teachingratings/replace_with_rvs_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Required columns for analysis
    required_cols = [
        'beauty', 'eval', 'students', 'age', 'gender', 'minority',
        'credits', 'division', 'native', 'tenure', 'prof'
    ]

    # Drop rows with missing values in any required columns
    df = df.dropna(subset=required_cols)

    # Standardize / clean categorical string columns (ensure consistent casing)
    for col in ['gender', 'minority', 'credits', 'division', 'native', 'tenure']:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Create binary / indicator control variables
    df['gender_female'] = (df['gender'] == 'female').astype(int)
    df['minority_yes'] = (df['minority'] == 'yes').astype(int)
    df['credits_single'] = (df['credits'] == 'single').astype(int)
    df['division_upper'] = (df['division'] == 'upper').astype(int)
    df['native_yes'] = (df['native'] == 'yes').astype(int)
    df['tenure_yes'] = (df['tenure'] == 'yes').astype(int)

    # Transform class size to log to reduce skew
    # Add a small constant if any zeros (shouldn't be the case here based on schema)
    df['log_students'] = np.log(df['students'].astype(float).clip(lower=1))

    # Center beauty (mean centering) to make intercept interpretable and reduce collinearity with interaction
    df['beauty_c'] = df['beauty'].astype(float) - float(df['beauty'].astype(float).mean())

    # Ensure eval is numeric
    df['eval'] = df['eval'].astype(float)

    # Final set of columns to keep (keeps original columns plus transformed ones)
    keep_cols = [
        'eval', 'beauty', 'beauty_c', 'gender_female', 'age', 'minority_yes',
        'credits_single', 'division_upper', 'native_yes', 'tenure_yes',
        'log_students', 'prof'
    ]

    # Some rows may have become invalid after transformations; drop NA in these essential columns
    df = df.dropna(subset=keep_cols)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fits an OLS model of teaching evaluations on centered beauty, a beauty x gender interaction,
    and controls. Clusters standard errors by instructor (prof).

    Returns:
        results: fitted statsmodels regression results instance (with clustered SEs applied)
    """
    import statsmodels.formula.api as smf

    # Formula: main effect of beauty (centered), interaction with gender, and controls
    formula = (
        'eval ~ beauty_c * gender_female + age + minority_yes + credits_single + '
        'division_upper + native_yes + tenure_yes + log_students'
    )

    # Fit OLS using formula API
    model = smf.ols(formula, data=df)

    # Fit and obtain clustered standard errors at the instructor (prof) level
    # If prof is numeric, it works for grouping. cov_type='cluster' uses cov_kwds={'groups': df['prof']}
    results = model.fit(cov_type='cluster', cov_kwds={'groups': df['prof']})

    # Return the fitted results object (has .summary(), .params, .bse, etc.)
    return results


