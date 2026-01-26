from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/replace_and_positive_statement_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Hamermesh classroom dataset into analysis-ready dataframe.

    Steps performed:
    - Drop rows missing the dependent variable (eval) or the primary predictor (beauty) or clustering variable (prof).
    - Standardize beauty (z-score) and create a quadratic term.
    - Create binary indicator control variables from categorical columns.
    - Compute log of students (number participating) to reduce skew.

    The returned dataframe contains at minimum the following columns (used in modeling):
    ['eval', 'beauty_z', 'beauty_sq', 'age', 'gender_female', 'minority_yes',
     'tenure_yes', 'native_yes', 'credits_single', 'division_lower', 'log_students', 'prof']
    """
    # Ensure we operate on a copy
    df = df.copy()

    # Drop rows missing key variables
    required_cols = ['eval', 'beauty', 'prof', 'students', 'age', 'gender', 'minority', 'tenure', 'native', 'credits', 'division']
    # If some columns are missing in df, only require the ones that exist; but here dataset includes them
    for c in ['eval', 'beauty', 'prof']:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in dataframe")
    df = df.dropna(subset=['eval', 'beauty', 'prof'])

    # Standardize beauty (z-score)
    beauty_mean = df['beauty'].mean()
    beauty_std = df['beauty'].std(ddof=0)
    # guard against zero-variance
    if beauty_std == 0 or np.isnan(beauty_std):
        df['beauty_z'] = 0.0
    else:
        df['beauty_z'] = (df['beauty'] - beauty_mean) / beauty_std
    df['beauty_sq'] = df['beauty_z'] ** 2

    # Numeric controls: keep age
    if 'age' in df.columns:
        # if age has missing values, keep them for now but model will drop rows with NA
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    else:
        df['age'] = np.nan

    # Binary indicators from categorical variables. Use explicit comparisons to avoid ordering issues.
    df['gender_female'] = (df['gender'].astype(str).str.lower() == 'female').astype(int) if 'gender' in df.columns else 0
    df['minority_yes'] = (df['minority'].astype(str).str.lower() == 'yes').astype(int) if 'minority' in df.columns else 0
    df['tenure_yes'] = (df['tenure'].astype(str).str.lower() == 'yes').astype(int) if 'tenure' in df.columns else 0
    df['native_yes'] = (df['native'].astype(str).str.lower() == 'yes').astype(int) if 'native' in df.columns else 0
    # credits: create indicator for 'single' credit course
    df['credits_single'] = (df['credits'].astype(str).str.lower() == 'single').astype(int) if 'credits' in df.columns else 0
    # division: indicator for lower division
    df['division_lower'] = (df['division'].astype(str).str.lower() == 'lower').astype(int) if 'division' in df.columns else 0

    # Log-transform students (participants); handle zeros/missing
    if 'students' in df.columns:
        # convert to numeric and drop/preserve smaller than 1 values by adding a tiny constant if necessary
        df['students'] = pd.to_numeric(df['students'], errors='coerce')
        df['students'] = df['students'].replace(0, np.nan)
        df['log_students'] = np.log(df['students'])
    else:
        df['log_students'] = np.nan

    # Ensure prof is present and convertible to int for clustering
    # Keep original prof values (use as grouping variable)
    df['prof'] = pd.to_numeric(df['prof'], errors='coerce')

    # Final: drop rows with NA in critical model columns to make modeling straightforward
    model_cols = ['eval', 'beauty_z', 'beauty_sq', 'age', 'gender_female', 'minority_yes',
                  'tenure_yes', 'native_yes', 'credits_single', 'division_lower', 'log_students', 'prof']
    df = df.dropna(subset=['eval', 'beauty_z', 'prof'])

    # Note: we do not force-drop rows with missing controls here; the model function will drop any remaining rows with NA
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression of student eval on instructor beauty (standardized) controlling for observable covariates.

    - Uses a quadratic term for beauty to allow non-linearity.
    - Clusters standard errors by `prof` (professor identifier) to account for multiple observations per professor.

    Returns the fitted statsmodels result object (with clustered covariance) and a tidy summary DataFrame.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Required columns in the transformed df
    required = ['eval', 'beauty_z', 'beauty_sq', 'age', 'gender_female', 'minority_yes',
                'tenure_yes', 'native_yes', 'credits_single', 'division_lower', 'log_students', 'prof']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Transformed dataframe is missing required columns: {missing}")

    # Drop rows with NA in any modeling column
    model_df = df[required].dropna()

    # Specify formula: main effects + quadratic
    formula = ('eval ~ beauty_z + beauty_sq + age + gender_female + minority_yes + '
               'tenure_yes + native_yes + credits_single + division_lower + log_students')

    # Fit OLS with clustering by professor id
    ols_fit = smf.ols(formula, data=model_df).fit(cov_type='cluster', cov_kwds={'groups': model_df['prof']})

    # Create a tidy summary table with coefficient, clustered SE, t, p, CI
    summary_frame = ols_fit.summary2().tables[1].copy()
    # summary2 with cluster cov_type already uses clustered se

    # Return the fitted model and the tidy coefficients table
    results = {
        'model_fit': ols_fit,
        'coef_table': summary_frame,
        'n_obs': int(model_df.shape[0])
    }
    return results


