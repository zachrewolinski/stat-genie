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
    """
    Clean and prepare the Hamermesh & Parker classroom beauty dataset for regression.

    Produces the following derived columns that are used in the model:
      - beauty_std: standardized beauty rating (mean 0, SD 1)
      - gender_female, minority_yes, tenure_yes, native_yes, credits_single, division_upper: binary dummies
      - log_students: log(students + 1)
      - prof: numeric professor id (kept for clustering)

    Keeps rows with non-missing eval and beauty and non-missing prof.
    """
    df = df.copy()

    # Ensure core numeric columns are numeric
    df['eval'] = pd.to_numeric(df['eval'], errors='coerce')
    df['beauty'] = pd.to_numeric(df['beauty'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['students'] = pd.to_numeric(df['students'], errors='coerce')
    df['prof'] = pd.to_numeric(df['prof'], errors='coerce')

    # Drop rows missing the dependent variable, the primary IV, or professor id (needed for clustering)
    df = df.dropna(subset=['eval', 'beauty', 'prof'])

    # Standardize beauty (use population-style divisor to be explicit)
    if df['beauty'].std(ddof=0) == 0 or np.isnan(df['beauty'].std(ddof=0)):
        df['beauty_std'] = 0.0
    else:
        df['beauty_std'] = (df['beauty'] - df['beauty'].mean()) / df['beauty'].std(ddof=0)

    # Binary indicators from categorical fields (coerce to lowercase strings to be robust)
    df['gender_female'] = df['gender'].astype(str).str.lower().map({'female': 1, 'male': 0})
    df['minority_yes'] = df['minority'].astype(str).str.lower().map({'yes': 1, 'no': 0})
    df['tenure_yes'] = df['tenure'].astype(str).str.lower().map({'yes': 1, 'no': 0})
    df['native_yes'] = df['native'].astype(str).str.lower().map({'yes': 1, 'no': 0})
    df['credits_single'] = df['credits'].astype(str).str.lower().map({'single': 1, 'more': 0})
    df['division_upper'] = df['division'].astype(str).str.lower().map({'upper': 1, 'lower': 0})

    # If any of the mappings produced NA (unexpected category values), fill with 0 (conservative)
    for col in ['gender_female', 'minority_yes', 'tenure_yes', 'native_yes', 'credits_single', 'division_upper']:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Fill missing students with the median to avoid dropping many rows; then take log
    if 'students' in df.columns:
        med_students = float(df['students'].median(skipna=True)) if df['students'].notna().any() else 0.0
        df['students'] = df['students'].fillna(med_students)
        # use +1 to avoid log(0)
        df['log_students'] = np.log(df['students'] + 1)
    else:
        # create default column if missing
        df['log_students'] = 0.0

    # Keep only columns needed for modeling plus original eval for interpretation
    keep_cols = [
        'eval', 'beauty', 'beauty_std', 'age', 'gender_female', 'minority_yes',
        'tenure_yes', 'native_yes', 'credits_single', 'division_upper', 'log_students', 'prof'
    ]

    # Add any missing keep_cols as NA-safe defaults
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model of teaching evaluations on standardized beauty and controls.

    Uses clustered robust standard errors at the professor level (prof) to account for
    within-instructor correlation across multiple course observations.

    Returns the fitted results object (statsmodels RegressionResultsWrapper).
    """
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['eval', 'beauty_std', 'age', 'gender_female', 'minority_yes',
                'tenure_yes', 'native_yes', 'credits_single', 'division_upper',
                'log_students', 'prof']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Specify the linear model. We include the main effect of beauty and standard controls.
    # If you want to test moderation by gender, add the interaction term: beauty_std:gender_female
    formula = (
        'eval ~ beauty_std + age + gender_female + minority_yes + tenure_yes + '
        'native_yes + credits_single + division_upper + log_students'
    )

    # Fit OLS and compute cluster-robust SEs by professor id
    mod = smf.ols(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prof']})

    # Return the fitted model object so the caller can inspect coefficients, summary, etc.
    return mod


