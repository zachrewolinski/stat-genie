from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/add_features_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Hamermesh & Parker classroom data into a modeling-ready dataframe.

    Produces the columns required by the model:
      - Eval: dependent variable (from 'eval')
      - Beauty_z: standardized beauty
      - Beauty_sq: squared standardized beauty
      - Gender_Female, age, Tenure_Yes, log_students, Division_upper, Credits_more,
        Native_English, Minority_Yes, religiousness, prof

    Notes:
      - Drops rows with missing values in key variables used in the model.
      - Log-transform class size ('students') to reduce skew.
    """
    df = df.copy()

    # Required raw columns for modeling
    required_cols = ['eval', 'beauty', 'gender', 'age', 'tenure', 'students',
                     'division', 'credits', 'native', 'minority', 'religiousness', 'prof']

    # Drop rows missing any required columns
    df = df.dropna(subset=required_cols)

    # Dependent variable
    df['Eval'] = df['eval'].astype(float)

    # Independent variable: standardize beauty (z-score)
    # Use sample mean/std from available rows
    beauty_mean = df['beauty'].mean()
    beauty_std = df['beauty'].std(ddof=0)
    if beauty_std == 0 or np.isnan(beauty_std):
        # fall back safe-guard
        df['Beauty_z'] = 0.0
    else:
        df['Beauty_z'] = (df['beauty'] - beauty_mean) / beauty_std

    # Quadratic term to capture nonlinearity
    df['Beauty_sq'] = df['Beauty_z'] ** 2

    # Moderator / control: Gender (female = 1, male = 0)
    df['Gender_Female'] = df['gender'].astype(str).str.lower().map(lambda x: 1 if x == 'female' else 0)

    # Age (keep numeric)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Tenure indicator
    df['Tenure_Yes'] = df['tenure'].astype(str).str.lower().map(lambda x: 1 if x == 'yes' else 0)

    # Log transform of students (number of respondents). students minimum is > 0 in this dataset.
    df['students'] = pd.to_numeric(df['students'], errors='coerce')
    # drop rows where students <= 0 or NaN after coercion
    df = df[df['students'] > 0]
    df['log_students'] = np.log(df['students'])

    # Division: upper = 1, lower = 0
    df['Division_upper'] = df['division'].astype(str).str.lower().map(lambda x: 1 if x == 'upper' else 0)

    # Credits: 'more' = 1, 'single' = 0
    df['Credits_more'] = df['credits'].astype(str).str.lower().map(lambda x: 1 if x == 'more' else 0)

    # Native English speaker
    df['Native_English'] = df['native'].astype(str).str.lower().map(lambda x: 1 if x == 'yes' else 0)

    # Minority indicator
    df['Minority_Yes'] = df['minority'].astype(str).str.lower().map(lambda x: 1 if x == 'yes' else 0)

    # Religiousness numeric (keep as-is but coerce)
    df['religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')

    # Instructor identifier for clustering
    # Ensure prof is integer-like
    df['prof'] = pd.to_numeric(df['prof'], errors='coerce')

    # Final drop: any remaining missing model columns
    model_cols = ['Eval', 'Beauty_z', 'Beauty_sq', 'Gender_Female', 'age', 'Tenure_Yes',
                  'log_students', 'Division_upper', 'Credits_more', 'Native_English',
                  'Minority_Yes', 'religiousness', 'prof']
    df = df.dropna(subset=model_cols)

    # Keep only columns needed for modeling (plus small set of originals if desired)
    keep_cols = model_cols
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model predicting course evaluation (Eval) from instructor beauty and controls.

    Model specification:
      Eval ~ Beauty_z + Beauty_sq + Gender_Female + Beauty_z:Gender_Female
           + age + Tenure_Yes + log_students + Division_upper + Credits_more
           + Native_English + Minority_Yes + religiousness

    Standard errors are clustered by instructor ('prof') to account for multiple courses taught
    by the same instructor.

    Returns the fitted statsmodels regression results object.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns present
    required = ['Eval', 'Beauty_z', 'Beauty_sq', 'Gender_Female', 'age', 'Tenure_Yes',
                'log_students', 'Division_upper', 'Credits_more', 'Native_English',
                'Minority_Yes', 'religiousness', 'prof']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula
    formula = (
        'Eval ~ Beauty_z + Beauty_sq + Gender_Female + Beauty_z:Gender_Female '
        '+ age + Tenure_Yes + log_students + Division_upper + Credits_more '
        '+ Native_English + Minority_Yes + religiousness'
    )

    # Fit OLS with clustered standard errors by prof
    model = smf.ols(formula=formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prof']})

    return model


