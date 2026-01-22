from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/teachingratings/negative_leading_statement_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for modeling.
    - Drops rows missing the dependent variable (eval) or the main independent variable (beauty).
    - Standardizes beauty to create beauty_z.
    - Creates binary indicator controls for categorical variables.
    - Creates a log transform of class size ('students') to reduce skew.
    - Ensures professor id is integer for clustering/fixed effects.

    Returns the dataframe with the columns used in the model: eval, beauty_z, age, is_female,
    is_minority, is_single_credit, is_upper_division, is_native, is_tenure, log_students, prof
    """
    df = df.copy()

    # Drop rows with missing dependent or main independent variables
    df = df.dropna(subset=['eval', 'beauty'])

    # Normalize categorical strings (defensive)
    for col in ['minority', 'gender', 'credits', 'division', 'native', 'tenure']:
        if col in df.columns:
            # convert to lowercase string to make mapping robust
            df[col] = df[col].astype(str).str.lower()

    # Binary controls (map to 1/0)
    # minority: 'yes' -> 1, 'no' -> 0
    if 'minority' in df.columns:
        df['is_minority'] = df['minority'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)
    else:
        df['is_minority'] = 0

    # gender: female -> 1, male -> 0
    if 'gender' in df.columns:
        df['is_female'] = df['gender'].map({'female': 1, 'male': 0}).fillna(0).astype(int)
    else:
        df['is_female'] = 0

    # credits: single -> 1, more -> 0
    if 'credits' in df.columns:
        df['is_single_credit'] = df['credits'].map({'single': 1, 'more': 0}).fillna(0).astype(int)
    else:
        df['is_single_credit'] = 0

    # division: upper -> 1, lower -> 0
    if 'division' in df.columns:
        df['is_upper_division'] = df['division'].map({'upper': 1, 'lower': 0}).fillna(0).astype(int)
    else:
        df['is_upper_division'] = 0

    # native English speaker: yes -> 1, no -> 0
    if 'native' in df.columns:
        df['is_native'] = df['native'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)
    else:
        df['is_native'] = 0

    # tenure track: yes -> 1, no -> 0
    if 'tenure' in df.columns:
        df['is_tenure'] = df['tenure'].map({'yes': 1, 'no': 0}).fillna(0).astype(int)
    else:
        df['is_tenure'] = 0

    # Log-transform the number of students who participated in the evaluation
    if 'students' in df.columns:
        # replace zeros or negative with NaN before log
        df['log_students'] = np.log(df['students'].replace({0: np.nan}))
    else:
        df['log_students'] = np.nan

    # Standardize beauty: mean 0, SD 1
    df['beauty_z'] = (df['beauty'] - df['beauty'].mean()) / (df['beauty'].std(ddof=0) if df['beauty'].std(ddof=0) != 0 else 1)

    # Ensure professor id is integer for clustering / fixed effects
    if 'prof' in df.columns:
        df['prof'] = df['prof'].astype(int)
    else:
        df['prof'] = -1

    # Keep only the variables needed for modeling to reduce memory
    keep_cols = ['eval', 'beauty_z', 'age', 'is_female', 'is_minority', 'is_single_credit',
                 'is_upper_division', 'is_native', 'is_tenure', 'log_students', 'prof']
    # If any keep_cols missing in df, they are already created above (with defaults)
    df = df[keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run two regression specifications to test whether instructor beauty affects student evaluations.

    Specification 1 (model1): OLS of eval on standardized beauty and controls, with standard errors
    clustered by professor to account for multiple courses per instructor.

    Specification 2 (model2): OLS including professor fixed effects (C(prof)) to control for time-invariant
    instructor unobservables (ability, reputation). Standard errors are clustered by professor.

    Returns a dictionary containing fitted results objects for both models.
    """
    import statsmodels.formula.api as smf

    # Drop any rows with missing values in predictors used in formulas
    df_model = df.dropna(subset=['eval', 'beauty_z', 'age', 'is_female', 'is_minority',
                                 'is_single_credit', 'is_upper_division', 'is_native',
                                 'is_tenure', 'log_students', 'prof'])

    # Base formula with controls
    formula = (
        'eval ~ beauty_z + age + is_female + is_minority + is_single_credit + '
        'is_upper_division + is_native + is_tenure + log_students'
    )

    # Model 1: OLS with clustered standard errors by professor
    model1 = smf.ols(formula, data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['prof']})

    # Model 2: add professor fixed effects to control for unobserved, time-invariant instructor characteristics
    # C(prof) creates dummy variables for each professor. We still cluster by prof.
    formula_fe = formula + ' + C(prof)'
    # To avoid perfect multicollinearity issues we rely on statsmodels to drop one category.
    model2 = smf.ols(formula_fe, data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['prof']})

    # Return both fitted results so the caller can inspect coefficients, p-values, and diagnostics
    return {
        'model1_clustered_by_prof': model1,
        'model2_prof_fixed_effects_clustered_by_prof': model2
    }


