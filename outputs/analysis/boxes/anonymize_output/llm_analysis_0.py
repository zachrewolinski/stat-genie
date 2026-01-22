from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/anonymize_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe suitable for modeling.

    Input columns used from raw df:
      - feature1: outcome (1=unchosen/undemonstrated option, 2=majority, 3=minority)
      - feature2: gender (1=girl, 2=boy)
      - feature3: age in years
      - feature4: whether majority was demonstrated first (0/1)
      - feature5: site id (integer)

    Output (columns created/kept):
      - Age (float)
      - Age_c (centered age)
      - AgeGroup (binned developmental stage)
      - Site (categorical string: 'Site_#')
      - IsBoy (0/1)
      - MajorityFirst (0/1)
      - SocialReliance (0/1)
      - MajorityChoice (0/1)
    """
    df = df.copy()

    # Keep rows with required fields
    df = df.dropna(subset=['feature1', 'feature3', 'feature5'])

    # Age
    df['Age'] = df['feature3'].astype(float)

    # Centered age for interpretability in interaction models
    df['Age_c'] = df['Age'] - df['Age'].mean()

    # Age groups for descriptive summaries / robustness checks (not required for main models)
    bins = [3, 6, 9, 12, 15]  # covers ages 4-14 into four bins
    labels = ['4-6', '7-9', '10-12', '13-14']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

    # Site / culture as categorical string to be used with C(Site) in formulas
    df['Site'] = df['feature5'].astype(int).astype(str)
    df['Site'] = 'Site_' + df['Site']

    # Gender: create binary IsBoy (1=boy, 0=girl). If feature2 has other codes or NaN, map conservatively.
    df['IsBoy'] = df['feature2'].map({1: 0, 2: 1}).fillna(0).astype(int)

    # Order: whether majority was shown first (assumed 0/1 already); ensure integer
    df['MajorityFirst'] = df['feature4'].astype(int)

    # Dependent variables derived from feature1
    # feature1: 1 = unchosen (undemonstrated), 2 = majority, 3 = minority
    df['SocialReliance'] = df['feature1'].apply(lambda x: 1 if x in [2, 3] else 0).astype(int)
    df['MajorityChoice'] = df['feature1'].apply(lambda x: 1 if x == 2 else 0).astype(int)

    # Final check: ensure dtypes are appropriate
    df['SocialReliance'] = df['SocialReliance'].astype(int)
    df['MajorityChoice'] = df['MajorityChoice'].astype(int)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit statistical models to answer whether reliance on social information and preference for majority cues
    vary across cultures and developmental stages.

    Models fit:
      1) Logistic regression for SocialReliance (binary) predicting whether child used social information.
         Formula: SocialReliance ~ Age_c * C(Site) + IsBoy + MajorityFirst
         The Age_c * C(Site) interaction tests whether age effects differ across sites (cultures).

      2) Logistic regression for MajorityChoice (binary), restricted to children who relied on social information
         (SocialReliance == 1). Formula: MajorityChoice ~ Age_c * C(Site) + IsBoy + MajorityFirst
         This tests whether preference for the majority among learners varies with age across sites.

    Returns a dict with model objects and sample sizes.
    """
    import statsmodels.formula.api as smf

    results = {}

    # Ensure required columns present
    required = ['SocialReliance', 'MajorityChoice', 'Age_c', 'Site', 'IsBoy', 'MajorityFirst']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Model 1: Social reliance (binary logistic regression)
    formula1 = 'SocialReliance ~ Age_c * C(Site) + IsBoy + MajorityFirst'
    try:
        social_model = smf.logit(formula1, data=df).fit(disp=False)
    except Exception as e:
        # If logistic fails (e.g., perfect separation), fall back to GLM with binomial family
        social_model = smf.glm(formula1, data=df, family=sm.families.Binomial()).fit()
    results['social_model'] = social_model

    # Model 2: Majority preference among those who used social information
    df_social = df[df['SocialReliance'] == 1].copy()
    results['n_total'] = int(df.shape[0])
    results['n_social_users'] = int(df_social.shape[0])

    if df_social.shape[0] < 10:
        results['majority_model'] = None
        results['majority_model_message'] = 'Too few cases who used social information to fit a stable model.'
    else:
        formula2 = 'MajorityChoice ~ Age_c * C(Site) + IsBoy + MajorityFirst'
        try:
            majority_model = smf.logit(formula2, data=df_social).fit(disp=False)
        except Exception as e:
            # fallback to GLM binomial when logit fails
            majority_model = smf.glm(formula2, data=df_social, family=sm.families.Binomial()).fit()
        results['majority_model'] = majority_model

    return results


