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
    """
    Transform the raw Hamermesh dataset into a modeling-ready dataframe.
    Outputs the columns used in the model:
      - Eval: dependent variable (same as 'eval')
      - beauty_std: standardized beauty score
      - IsFemale, IsMinority, IsTenure, IsNative, IsSingleCredit, IsUpperDivision: binary controls
      - Age_z, Students_z: standardized continuous controls
      - prof: professor id (kept for clustering)
    """
    # Work on a copy
    df = df.copy()

    # Drop rows missing key variables: beauty, eval, and core covariates used below
    required_cols = [
        'beauty', 'eval', 'gender', 'minority', 'tenure', 'native',
        'credits', 'division', 'age', 'students', 'prof'
    ]
    df = df.dropna(subset=required_cols)

    # Dependent variable: keep original eval but rename for clarity
    df['Eval'] = df['eval'].astype(float)

    # Independent variable: standardized beauty
    df['beauty_std'] = (df['beauty'] - df['beauty'].mean()) / df['beauty'].std(ddof=0)

    # Binary control variables (explicit mappings)
    df['IsFemale'] = (df['gender'].astype(str).str.lower() == 'female').astype(int)
    df['IsMinority'] = (df['minority'].astype(str).str.lower() == 'yes').astype(int)
    df['IsTenure'] = (df['tenure'].astype(str).str.lower() == 'yes').astype(int)
    df['IsNative'] = (df['native'].astype(str).str.lower() == 'yes').astype(int)
    df['IsSingleCredit'] = (df['credits'].astype(str).str.lower() == 'single').astype(int)
    df['IsUpperDivision'] = (df['division'].astype(str).str.lower() == 'upper').astype(int)

    # Standardize continuous covariates for interpretability
    df['Age_z'] = (df['age'] - df['age'].mean()) / df['age'].std(ddof=0)
    df['Students_z'] = (df['students'] - df['students'].mean()) / df['students'].std(ddof=0)

    # Ensure professor id is integer (used for clustering)
    df['prof'] = df['prof'].astype(int)

    # Final: keep only columns needed for modeling to avoid accidental usage of others
    keep_cols = [
        'Eval', 'beauty_std', 'IsFemale', 'IsMinority', 'IsTenure', 'IsNative',
        'IsSingleCredit', 'IsUpperDivision', 'Age_z', 'Students_z', 'prof'
    ]
    df = df.loc[:, keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model estimating the effect of standardized beauty (beauty_std) on teaching evaluations (Eval).
    The model includes covariates and an interaction testing whether the beauty effect differs by instructor gender.
    Cluster-robust standard errors are computed at the professor level ('prof').

    Model formula:
      Eval ~ beauty_std * IsFemale + IsMinority + IsTenure + IsNative + IsSingleCredit + IsUpperDivision + Age_z + Students_z

    Returns the fitted results object (statsmodels RegressionResultsWrapper) with clustered SEs.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['Eval', 'beauty_std', 'IsFemale', 'IsMinority', 'IsTenure', 'IsNative',
                'IsSingleCredit', 'IsUpperDivision', 'Age_z', 'Students_z', 'prof']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe required for modeling: {missing}")

    formula = (
        'Eval ~ beauty_std * IsFemale + IsMinority + IsTenure + IsNative '
        '+ IsSingleCredit + IsUpperDivision + Age_z + Students_z'
    )

    mod = smf.ols(formula=formula, data=df)
    # Fit with cluster-robust SEs by professor id
    results = mod.fit(cov_type='cluster', cov_kwds={'groups': df['prof']})

    return results


