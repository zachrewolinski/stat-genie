from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/campus/austin.zane/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/boxes/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to the analysis dataframe.

    Produces the following columns used in modeling:
      - MajorityChoice: binary (1 if chosen majority option, 0 otherwise)
      - AgeYears: numeric age in years (copied from 'culture' column in provided schema)
      - Age_c: age centered (AgeYears - mean(AgeYears))
      - Age_z: standardized age (z-score)
      - Gender_Male: 1 if gender==2 (boy), 0 if gender==1 (girl)
      - MajorityFirstDemo: indicator from the 'age' column in the schema which encodes whether majority was demonstrated first (0/1)
      - Site: categorical site ID derived from column 'y'

    Notes: the provided schema has somewhat confusing column labels (e.g., 'culture' contains age values and 'age' contains a 0/1 indicator for demonstration order). The transform below follows that mapping.
    """
    df = df.copy()

    # Ensure numeric columns where expected and drop rows with missing critical variables
    for col in ['majority_first', 'culture', 'gender', 'age', 'y']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the essential variables for the analysis
    df = df.dropna(subset=['majority_first', 'culture', 'gender', 'age', 'y'])

    # Keep only valid response codes for majority_first (expected 1,2,3 per schema)
    df = df[df['majority_first'].isin([1, 2, 3])].copy()

    # Dependent variable: did the child choose the majority-demonstrated option?
    # Schema: majority_first: 1 = unchosen option, 2 = majority option, 3 = minority option
    df['MajorityChoice'] = (df['majority_first'] == 2).astype(int)

    # Independent variable: age in years is provided in the 'culture' column per schema
    df['AgeYears'] = df['culture'].astype(float)
    # Center and standardize age for modeling
    age_mean = df['AgeYears'].mean()
    age_std = df['AgeYears'].std(ddof=0) if df['AgeYears'].std(ddof=0) != 0 else 1.0
    df['Age_c'] = df['AgeYears'] - age_mean
    df['Age_z'] = (df['AgeYears'] - age_mean) / age_std

    # Control: gender (schema: 1 = girl, 2 = boy). Create male indicator
    df['Gender_Male'] = (df['gender'] == 2).astype(int)

    # Control: whether majority was demonstrated first. In the provided schema this is the 'age' column (0/1)
    df['MajorityFirstDemo'] = df['age'].astype(int)

    # Site / cultural context: use 'y' column as categorical site identifier
    df['Site'] = df['y'].astype(int).astype('category')

    # Return a dataframe containing columns required for modeling (plus some originals for traceability)
    cols_out = [
        'MajorityChoice',
        'AgeYears',
        'Age_c',
        'Age_z',
        'Gender_Male',
        'MajorityFirstDemo',
        'Site',
        # keep original identifier columns for possible downstream checks
        'majority_first',
        'culture',
        'gender',
        'age',
        'y'
    ]

    # If any of these original columns are missing in input, adjust to available
    cols_out = [c for c in cols_out if c in df.columns]

    return df[cols_out]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a binomial (logistic) GLM predicting majority choice from age and its interaction with site (culture),
    controlling for gender and demonstration order. Returns the fitted GLMResults object.

    Model formula:
      MajorityChoice ~ Age_c * C(Site) + Gender_Male + MajorityFirstDemo

    We fit a GLM with Binomial family and compute cluster-robust standard errors clustered on Site.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required = ['MajorityChoice', 'Age_c', 'Site', 'Gender_Male', 'MajorityFirstDemo']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula: allow Age x Site interaction to test whether age trajectories differ across sites
    formula = 'MajorityChoice ~ Age_c * C(Site) + Gender_Male + MajorityFirstDemo'

    # Fit GLM (binomial)
    model_glm = smf.glm(formula=formula, data=df, family=sm.families.Binomial())

    # Fit and compute cluster-robust SEs clustered by Site (useful because observations within sites may be correlated)
    try:
        results = model_glm.fit(cov_type='cluster', cov_kwds={'groups': df['Site']})
    except Exception:
        # Fallback to basic fit if clustered cov fails for some reason
        results = model_glm.fit()

    return results


