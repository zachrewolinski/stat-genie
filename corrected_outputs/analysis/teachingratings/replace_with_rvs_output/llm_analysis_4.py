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
    Transform the raw Hamermesh dataset to create analytic variables.

    Adds standardized beauty (beauty_z), a quadratic term (beauty_sq),
    an interaction of beauty with gender (beauty_gender), dummies for
    categorical controls, standardized age (age_z), and log of students (log_students).

    Returns the dataframe with added columns used in the model.
    """
    df = df.copy()

    # Keep rows with required numeric outcomes/predictors
    df = df.dropna(subset=['eval', 'beauty'])

    # Standardize beauty and age (z-scores). Use population std (ddof=0) for stability.
    df['beauty_z'] = (df['beauty'] - df['beauty'].mean()) / df['beauty'].std(ddof=0)
    df['beauty_sq'] = df['beauty_z'] ** 2

    # Create gender dummy (female = 1, male = 0). Normalize strings to lowercase.
    df['gender_female'] = df['gender'].astype(str).str.lower().map({'female': 1, 'male': 0})
    df['gender_female'] = df['gender_female'].fillna(0).astype(int)

    # Interaction between standardized beauty and female indicator
    df['beauty_gender'] = df['beauty_z'] * df['gender_female']

    # Other binary controls: map yes/no to 1/0 and fill NAs with 0 (conservative)
    df['minority_yes'] = df['minority'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)
    df['tenure_yes'] = df['tenure'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)
    df['native_yes'] = df['native'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)
    df['division_upper'] = df['division'].astype(str).str.lower().map({'upper': 1, 'lower': 0}).fillna(0).astype(int)
    df['credits_single'] = df['credits'].astype(str).str.lower().map({'single': 1, 'more': 0}).fillna(0).astype(int)

    # Standardize age
    df['age_z'] = (df['age'] - df['age'].mean()) / df['age'].std(ddof=0)

    # Log-transform students who participated in the evaluation to reduce skew.
    # Replace zeros with NaN (there are none expected), and drop resulting NaNs later.
    df['log_students'] = np.log(df['students'].replace({0: np.nan}))

    # Ensure prof column exists for clustering (keep original values)
    df['prof'] = df['prof']

    # Drop any rows that lost information (e.g., missing students -> cannot compute log)
    df = df.dropna(subset=['log_students'])

    # Final check: keep only columns required for modeling (and original eval/prof)
    # but return full dataframe (as caller may inspect other variables). Required columns are created above.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model of teaching evaluations on beauty and controls.

    The specification includes:
      - beauty_z: standardized beauty (linear effect)
      - beauty_sq: quadratic beauty term to probe nonlinearity
      - beauty_gender: interaction of beauty_z with being female
      - gender_female and other demographic/course controls

    Standard errors are clustered by professor (prof) to account for
    within-instructor correlation across courses.

    Returns the fitted results object (statsmodels RegressionResultsWrapper).
    """
    # Columns used as regressors
    X_cols = [
        'beauty_z',
        'beauty_sq',
        'beauty_gender',
        'gender_female',
        'age_z',
        'minority_yes',
        'tenure_yes',
        'native_yes',
        'division_upper',
        'credits_single',
        'log_students'
    ]

    # Ensure required columns are present
    missing = [c for c in X_cols + ['eval', 'prof'] if c not in df.columns]
    if missing:
        raise ValueError('Dataframe is missing required columns: ' + ', '.join(missing))

    X = df[X_cols]
    X = sm.add_constant(X, has_constant='add')
    y = df['eval']

    ols_model = sm.OLS(y, X)

    # Fit with cluster-robust SEs by professor
    results = ols_model.fit(cov_type='cluster', cov_kwds={'groups': df['prof']})

    # Print summary for immediate inspection; return results for programmatic use
    print(results.summary())
    return results


