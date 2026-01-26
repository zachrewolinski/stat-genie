from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/shuffle_names_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modeling.

    Inputs expected (columns from provided dataset):
      - prof: continuous beauty rating (mean-zero in original, but we will standardize)
      - tenure: course evaluation score (DV)
      - division: instructor age (numeric)
      - age: instructor gender (strings like 'male'/'female')
      - students: course division ('lower'/'upper')
      - allstudents: indicator whether instructor is native English speaker ('yes'/'no')
      - eval: indicator whether instructor is on tenure track ('yes'/'no')
      - minority: number of students participating in evaluation
      - credits: number of students enrolled
      - rownames: instructor identifier

    Returns dataframe containing columns used in the model.
    """
    df = df.copy()

    # Ensure key numeric columns are numeric
    df['prof'] = pd.to_numeric(df['prof'], errors='coerce')
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
    df['instructor_age'] = pd.to_numeric(df['division'], errors='coerce')
    df['n_students'] = pd.to_numeric(df['minority'], errors='coerce')
    df['n_enrolled'] = pd.to_numeric(df['credits'], errors='coerce')

    # Basic required-case filtering: must have DV and IV
    df = df.dropna(subset=['prof', 'tenure'])

    # Normalize / standardize beauty rating for interpretability
    # Use population std (ddof=0) to match many typical standardization approaches
    if df['prof'].std(ddof=0) == 0 or np.isnan(df['prof'].std(ddof=0)):
        df['prof_z'] = df['prof'] - df['prof'].mean()
    else:
        df['prof_z'] = (df['prof'] - df['prof'].mean()) / df['prof'].std(ddof=0)

    # Gender: dataset column 'age' contains gender labels per schema; normalize strings
    df['gender_raw'] = df['age'].astype(str).str.lower().str.strip()
    df['gender_male'] = np.where(df['gender_raw'] == 'male', 1, 0)

    # Course division (lower/upper)
    df['course_lower'] = np.where(df['students'].astype(str).str.lower().str.strip() == 'lower', 1, 0)

    # Native English speaker indicator
    df['native_english'] = np.where(df['allstudents'].astype(str).str.lower().str.strip() == 'yes', 1, 0)

    # Tenure-track indicator
    df['tenure_track'] = np.where(df['eval'].astype(str).str.lower().str.strip() == 'yes', 1, 0)

    # Log transforms of counts to reduce skew
    df['log_n_students'] = np.log1p(df['n_students'])
    df['log_n_enrolled'] = np.log1p(df['n_enrolled'])

    # Instructor id used for clustering (keep original 'rownames' as identifier)
    # Ensure it's present and numeric if possible
    df['instructor_id'] = df['rownames']

    # Drop observations with missing values in key controls (we've already dropped missing IV/DV)
    df = df.dropna(subset=['instructor_age', 'n_students', 'n_enrolled', 'instructor_id'])

    # Keep only the columns required for modeling to make output compact
    keep_cols = [
        'prof', 'prof_z', 'tenure', 'gender_male', 'instructor_age', 'course_lower',
        'native_english', 'tenure_track', 'n_students', 'n_enrolled', 'log_n_students',
        'log_n_enrolled', 'instructor_id'
    ]
    # Some columns may not exist in edge cases, so intersect with available
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS regression of course evaluation (tenure) on standardized beauty (prof_z)
    with controls and an interaction with gender. Cluster standard errors by instructor.

    Model formula:
      tenure ~ prof_z + gender_male + instructor_age + course_lower + native_english
               + tenure_track + log_n_students + log_n_enrolled + prof_z:gender_male

    Returns the fitted statsmodels results object (OLS with clustered SEs).
    """
    import statsmodels.formula.api as smf

    # Work on a copy to avoid modifying caller data
    df = df.copy()

    # Ensure required columns are present
    required = ['tenure', 'prof_z', 'gender_male', 'instructor_age', 'course_lower',
                'native_english', 'tenure_track', 'log_n_students', 'log_n_enrolled', 'instructor_id']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for model: {missing}")

    formula = (
        'tenure ~ prof_z + gender_male + instructor_age + course_lower + native_english '
        '+ tenure_track + log_n_students + log_n_enrolled + prof_z:gender_male'
    )

    # Fit OLS
    mod = smf.ols(formula=formula, data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['instructor_id']}
    )

    # Return the fitted results object (caller can inspect .summary() or coefficients)
    return mod


