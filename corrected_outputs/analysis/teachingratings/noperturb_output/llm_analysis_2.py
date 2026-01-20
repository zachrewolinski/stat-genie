from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/teachingratings/noperturb_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for OLS regression of eval on beauty and controls.

    Transformations performed:
    - Drop rows missing any variables required for the model.
    - Standardize beauty to create 'beauty_z'.
    - Create binary dummies for categorical controls: gender_female, minority_yes,
      tenure_yes, native_yes, credits_single, division_upper.
    - Create students_log = log(students).
    - Ensure 'prof' is present for clustering.

    Returns a dataframe with the new columns used in the model.
    """
    df = df.copy()

    # Columns required for the planned model
    required_cols = [
        'eval', 'beauty', 'age', 'gender', 'minority', 'tenure', 'native',
        'credits', 'division', 'students', 'prof'
    ]

    # Drop rows missing any of the required columns
    df = df.dropna(subset=required_cols)

    # Ensure numeric types where applicable
    df['eval'] = pd.to_numeric(df['eval'], errors='coerce')
    df['beauty'] = pd.to_numeric(df['beauty'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['students'] = pd.to_numeric(df['students'], errors='coerce')
    df['prof'] = pd.to_numeric(df['prof'], errors='coerce')

    # Drop again rows that became NaN after coercion
    df = df.dropna(subset=['eval', 'beauty', 'age', 'students', 'prof'])

    # Standardize beauty (z-score)
    beauty_mean = df['beauty'].mean()
    beauty_std = df['beauty'].std(ddof=0) if df['beauty'].std(ddof=0) != 0 else 1.0
    df['beauty_z'] = (df['beauty'] - beauty_mean) / beauty_std

    # Normalize categorical text to lower-case strings to avoid mismatches
    for col in ['gender', 'minority', 'tenure', 'native', 'credits', 'division']:
        # Only operate if column exists and is object-like
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    # Create binary dummies
    df['gender_female'] = (df['gender'] == 'female').astype(int)
    df['minority_yes'] = (df['minority'] == 'yes').astype(int)
    df['tenure_yes'] = (df['tenure'] == 'yes').astype(int)
    df['native_yes'] = (df['native'] == 'yes').astype(int)
    df['credits_single'] = (df['credits'] == 'single').astype(int)
    df['division_upper'] = (df['division'] == 'upper').astype(int)

    # Log transform number of students to reduce skew
    # Use natural log; students reported >=5, so log defined
    df['students_log'] = np.log(df['students'].replace(0, np.nan))

    # Final drop for any rows that might have resulted in NaN in derived columns
    model_cols = [
        'eval', 'beauty_z', 'age', 'gender_female', 'minority_yes', 'tenure_yes',
        'native_yes', 'credits_single', 'division_upper', 'students_log', 'prof'
    ]
    df = df.dropna(subset=model_cols)

    # Keep only the columns needed for modeling plus the original eval and beauty
    keep_cols = ['eval', 'beauty', 'beauty_z', 'age', 'gender_female', 'minority_yes',
                 'tenure_yes', 'native_yes', 'credits_single', 'division_upper',
                 'students_log', 'prof']
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit OLS regression of evaluation score on standardized beauty and controls.

    Model specification:
      eval ~ beauty_z + age + gender_female + minority_yes + tenure_yes
             + native_yes + credits_single + division_upper + students_log

    Standard errors are clustered at the professor level ('prof') to account for
    multiple course observations per instructor.

    Returns the fitted statsmodels RegressionResults object.
    """
    import statsmodels.formula.api as smf

    # Ensure df contains columns expected by the model
    required = ['eval', 'beauty_z', 'age', 'gender_female', 'minority_yes',
                'tenure_yes', 'native_yes', 'credits_single', 'division_upper',
                'students_log', 'prof']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula
    formula = (
        'eval ~ beauty_z + age + gender_female + minority_yes + tenure_yes '
        '+ native_yes + credits_single + division_upper + students_log'
    )

    # Fit OLS with clustered standard errors by professor
    model = smf.ols(formula=formula, data=df).fit(
        cov_type='cluster',
        cov_kwds={'groups': df['prof']}
    )

    # Return the fitted model (RegressionResults) so user can inspect summary, params, etc.
    return model


