from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/teachingratings/add_features_output/teachingratings.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Required raw columns for the planned model
    required = ['beauty', 'eval', 'age', 'students', 'religiousness', 'gender', 'tenure', 'division', 'credits', 'minority', 'native', 'prof']
    df = df.dropna(subset=required)

    # Standardize the beauty measure for interpretation
    df['beauty_z'] = (df['beauty'] - df['beauty'].mean()) / df['beauty'].std()
    df['beauty_z_sq'] = df['beauty_z'] ** 2

    # Transform skewed enrollment variable
    # students is >=5 in this dataset, so log is well-defined
    df['students_log'] = np.log(df['students'])

    # Create binary indicators from categorical factors (map expected levels; lower-case to be robust)
    df['gender_F'] = df['gender'].astype(str).str.lower().map({'female': 1, 'male': 0})
    df['tenure_yes'] = df['tenure'].astype(str).str.lower().map({'yes': 1, 'no': 0})
    df['division_upper'] = df['division'].astype(str).str.lower().map({'upper': 1, 'lower': 0})
    df['credits_more'] = df['credits'].astype(str).str.lower().map({'more': 1, 'single': 0})
    df['minority_yes'] = df['minority'].astype(str).str.lower().map({'yes': 1, 'no': 0})
    df['native_yes'] = df['native'].astype(str).str.lower().map({'yes': 1, 'no': 0})

    # Drop any rows that failed mapping (unexpected categories)
    mapped_cols = ['gender_F', 'tenure_yes', 'division_upper', 'credits_more', 'minority_yes', 'native_yes']
    df = df.dropna(subset=mapped_cols)

    # Ensure prof is integer for clustering
    df['prof'] = df['prof'].astype(int)

    # Keep only the columns needed for modeling (plus originals for traceability)
    keep_cols = ['beauty', 'beauty_z', 'beauty_z_sq', 'eval', 'age', 'students', 'students_log', 'religiousness',
                 'gender_F', 'tenure_yes', 'division_upper', 'credits_more', 'minority_yes', 'native_yes', 'prof']
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # Statistical model: OLS of evaluation on beauty (linear + quadratic) with controls.
    # Clustered (robust) standard errors at the instructor (prof) level.
    import statsmodels.formula.api as smf

    formula = (
        'eval ~ beauty_z + beauty_z_sq + age + students_log + religiousness + '
        'gender_F + tenure_yes + division_upper + credits_more + minority_yes + native_yes'
    )

    mod = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prof']})

    # Return the fitted results object (has .summary(), .params, .bse, etc.)
    return mod


