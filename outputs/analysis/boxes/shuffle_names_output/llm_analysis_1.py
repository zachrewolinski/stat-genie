from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/shuffle_names_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis-ready dataframe.

    Outputs the following columns (these exact names are required by the model function):
      - majority_first: original response (kept for traceability)
      - AgeYears: continuous age in years (from 'culture' column in the raw data)
      - AgeGroup: categorical developmental stage (4-6, 7-9, 10-12, 13-14)
      - Site: categorical site ID (from 'y')
      - Gender: numeric coded 0 = girl, 1 = boy
      - MajorityDemonstratedFirst: 0/1 whether majority was demonstrated first (from 'age' column)
      - SocialUse: binary 0/1: 1 if child chose a demonstrated option (majority or minority), else 0
      - MajorityPreference: binary 0/1: 1 if child chose the majority option, else 0
    """
    # Work on a copy
    df = df.copy()

    # Drop rows missing essential values
    required_cols = ['majority_first', 'culture', 'gender', 'age', 'y']
    df = df.dropna(subset=required_cols)

    # Age: in this dataset the 'culture' column holds age-in-years (per provided schema)
    df['AgeYears'] = pd.to_numeric(df['culture'], errors='coerce')

    # Create AgeGroup categorical bins to capture non-linear developmental stages
    # bins chosen to reflect early childhood, middle childhood, later childhood, early adolescence
    bins = [3, 6, 9, 12, 15]  # edges: (3,6], (6,9], (9,12], (12,15]
    labels = ['4-6', '7-9', '10-12', '13-14']
    df['AgeGroup'] = pd.cut(df['AgeYears'], bins=bins, labels=labels, right=True)

    # Site: use the 'y' column as site identifier
    df['Site'] = df['y'].astype('category')

    # Gender: map 1 -> girl (0), 2 -> boy (1)
    df['Gender'] = pd.to_numeric(df['gender'], errors='coerce')
    df['Gender'] = df['Gender'].map({1: 0, 2: 1})

    # MajorityDemonstratedFirst: provided in 'age' column per schema (0/1 indicator)
    df['MajorityDemonstratedFirst'] = pd.to_numeric(df['age'], errors='coerce')

    # Dependent variables
    # majority_first codes: 1 = undemonstrated (unchosen option), 2 = majority option, 3 = minority option
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce')

    # SocialUse: used social information (picked majority OR minority)
    df['SocialUse'] = df['majority_first'].apply(lambda x: 1 if x in [2, 3] else 0)

    # MajorityPreference: picked the majority demonstrated option
    df['MajorityPreference'] = df['majority_first'].apply(lambda x: 1 if x == 2 else 0)

    # Final drop of any rows that still have NA in the columns we'll model on
    final_cols = ['majority_first', 'AgeYears', 'AgeGroup', 'Site', 'Gender',
                  'MajorityDemonstratedFirst', 'SocialUse', 'MajorityPreference']
    df = df.dropna(subset=final_cols)

    # Cast to stable numpy dtypes that patsy/statsmodels can handle (avoid pandas nullable dtypes)
    # AgeYears: keep as float
    df['AgeYears'] = df['AgeYears'].astype(float)

    # AgeGroup and Site as categorical
    df['AgeGroup'] = df['AgeGroup'].astype('category')
    df['Site'] = df['Site'].astype('category')

    # Binary / integer columns to plain numpy int64
    int_cols = ['majority_first', 'Gender', 'MajorityDemonstratedFirst', 'SocialUse', 'MajorityPreference']
    for col in int_cols:
        # safe conversion: values are guaranteed non-null due to dropna above
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.int64)

    # Return only the columns needed for modeling (keeps traceability column majority_first)
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two logistic (binomial) models to answer the research question:
      1) Reliance on social information (SocialUse): whether children pick a demonstrated option (majority or minority) vs the undemonstrated option.
      2) Majority preference among social learners (MajorityPreference): whether children who used social information picked the majority option.

    Both models use AgeYears (continuous) and Site (categorical) as primary predictors and include Gender and MajorityDemonstratedFirst as controls.
    We include an interaction AgeYears * Site to test whether developmental trajectories differ across sites (cultures).

    Returns a dict with fitted model result objects and printed summaries.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure Site is treated as categorical
    df = df.copy()
    df['Site'] = df['Site'].astype('category')

    # Model 1: SocialUse (binary) ~ AgeYears * Site + Gender + MajorityDemonstratedFirst
    formula1 = 'SocialUse ~ AgeYears * C(Site) + Gender + MajorityDemonstratedFirst'
    model1 = smf.glm(formula=formula1, data=df, family=sm.families.Binomial())

    # Fit with cluster-robust SE clustered by Site (helps account for within-site correlation)
    try:
        res1 = model1.fit(cov_type='cluster', cov_kwds={'groups': df['Site']})
    except Exception:
        # fallback to default fit if clustering fails
        res1 = model1.fit()

    print('Model 1: Reliance on social information (SocialUse)')
    print(res1.summary())

    # Model 2: MajorityPreference among social learners only
    df_social = df[df['SocialUse'] == 1].copy()

    # If there are too few social learners in a particular site, the interaction may be unstable.
    # Formula is the same structure but fit on the restricted dataset
    formula2 = 'MajorityPreference ~ AgeYears * C(Site) + Gender + MajorityDemonstratedFirst'
    model2 = smf.glm(formula=formula2, data=df_social, family=sm.families.Binomial())

    try:
        res2 = model2.fit(cov_type='cluster', cov_kwds={'groups': df_social['Site']})
    except Exception:
        res2 = model2.fit()

    print('\nModel 2: Majority preference among social learners (MajorityPreference)')
    print(res2.summary())

    # Return the results objects so the caller can examine coefficients, confidence intervals, predicted values, etc.
    return {
        'model_social_use': res1,
        'model_majority_pref': res2,
        'df_used_for_model_1_shape': df.shape,
        'df_used_for_model_2_shape': df_social.shape
    }