from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/replace_with_rvs_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for modeling:
    - Drop rows missing the dependent variable (eval), the main independent variable (beauty), or required controls.
    - Create binary indicator controls from categorical fields.
    - Center beauty and add quadratic term to allow non-linear effect.
    - Log-transform students (number of respondents) to reduce skew.

    Returns a dataframe that contains all columns named in the conceptual variables.
    """

    # Make a copy to avoid modifying the original passed DF
    df = df.copy()

    # Ensure required columns exist
    required_cols = [
        'eval', 'beauty', 'gender', 'age', 'minority', 'credits',
        'division', 'native', 'tenure', 'students', 'prof'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for transform: {missing}")

    # Drop rows with NA in key modeling columns
    df = df.dropna(subset=['eval', 'beauty', 'gender', 'age', 'students', 'prof'])

    # Encode categorical controls as binary indicators
    # Note: we treat the string values exactly as in the schema (e.g., 'female', 'yes', 'single', 'upper')
    df['gender_female'] = (df['gender'] == 'female').astype(int)
    df['minority_yes'] = (df['minority'] == 'yes').astype(int)
    df['credits_single'] = (df['credits'] == 'single').astype(int)
    df['division_upper'] = (df['division'] == 'upper').astype(int)
    df['native_yes'] = (df['native'] == 'yes').astype(int)
    df['tenure_yes'] = (df['tenure'] == 'yes').astype(int)

    # Numeric controls: convert to numeric types if necessary
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['students'] = pd.to_numeric(df['students'], errors='coerce')

    # After conversions, drop rows where numeric controls became NA
    df = df.dropna(subset=['age', 'students'])

    # Center beauty around its sample mean and add squared term for nonlinearity
    beauty_mean = df['beauty'].mean()
    df['beauty_c'] = df['beauty'] - beauty_mean
    df['beauty_sq'] = df['beauty_c'] ** 2

    # Log-transform number of students to reduce skew (add small constant for safety)
    df['students_log'] = np.log(df['students'].clip(lower=1))

    # Keep only columns needed for modeling plus any others the user might want
    required_model_cols = [
        'eval', 'beauty', 'beauty_c', 'beauty_sq',
        'gender_female', 'age', 'minority_yes', 'credits_single',
        'division_upper', 'native_yes', 'tenure_yes', 'students_log', 'prof'
    ]

    # If additional columns exist, leave them; return DF with all transformed columns present
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model of student evaluations on instructor beauty and controls.

    Model specification:
      eval ~ beauty_c + beauty_sq + gender_female + beauty_c:gender_female
           + age + minority_yes + credits_single + division_upper + native_yes + tenure_yes
           + students_log + C(prof)

    We include C(prof) (instructor fixed effects) to absorb unobserved, time-invariant instructor characteristics.
    Standard errors are clustered by instructor ('prof') to account for non-independence of observations for the same instructor.

    Returns the robust (clustered) results object.
    """
    import statsmodels.formula.api as smf

    # Verify required transformed columns exist
    required = [
        'eval', 'beauty_c', 'beauty_sq', 'gender_female', 'age', 'minority_yes',
        'credits_single', 'division_upper', 'native_yes', 'tenure_yes', 'students_log', 'prof'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # Define formula with an interaction between beauty and gender
    formula = (
        'eval ~ beauty_c + beauty_sq + gender_female + beauty_c:gender_female '
        '+ age + minority_yes + credits_single + division_upper + native_yes + tenure_yes '
        '+ students_log + C(prof)'
    )

    # Fit OLS via formula API
    ols_res = smf.ols(formula, data=df).fit()

    # Compute cluster-robust standard errors clustered on instructor id (prof)
    # If 'prof' has many unique values clustering is appropriate; ensure it's integer or categorical
    try:
        clustered_res = ols_res.get_robustcov_results(cov_type='cluster', groups=df['prof'])
    except Exception:
        # Fall back to heteroskedasticity-robust (HC3) if clustering fails
        clustered_res = ols_res.get_robustcov_results(cov_type='HC3')

    # Print summary and return the robust-results object for further inspection
    print(clustered_res.summary())
    return clustered_res


