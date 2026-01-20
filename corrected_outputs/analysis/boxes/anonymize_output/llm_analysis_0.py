from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/boxes/anonymize_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready DataFrame with the following columns used in modeling:
      - MajorityChoice: binary DV (1 if feature1 == 2 [majority], else 0)
      - AgeYears: original age in years (feature3)
      - Age_c: mean-centered age
      - Age_c2: squared centered age for nonlinear effects
      - Site: categorical site identifier (feature5 cast to string)
      - Female: gender binary (1 if feature2 == 1 (girl), 0 if feature2 == 2 (boy))
      - MajorityFirst: whether the majority was demonstrated first (feature4 as 0/1)
    """
    # Work on a copy
    df = df.copy()

    # Drop rows with missing essential fields
    required = ['feature1', 'feature3', 'feature5', 'feature2', 'feature4']
    df = df.dropna(subset=required)

    # Dependent variable: majority choice (feature1: 2 == majority)
    df['MajorityChoice'] = (df['feature1'] == 2).astype(int)

    # Age variables
    df['AgeYears'] = pd.to_numeric(df['feature3'], errors='coerce')
    # If AgeYears produced any NA (shouldn't after dropna) keep them dropped
    df = df.dropna(subset=['AgeYears'])
    df['Age_c'] = df['AgeYears'] - df['AgeYears'].mean()
    df['Age_c2'] = df['Age_c'] ** 2

    # Site as categorical (cast to string so C(Site) in formula treats it categorically)
    df['Site'] = df['feature5'].astype(int).astype(str)

    # Gender: feature2: 1 = girl, 2 = boy. Create Female = 1 for girls, 0 for boys
    df['Female'] = df['feature2'].apply(lambda x: 1 if int(x) == 1 else 0)

    # Whether majority was demonstrated first: feature4 (0/1 already)
    df['MajorityFirst'] = df['feature4'].astype(int)

    # Keep only the columns needed for modeling (and any identifiers if desired)
    final_cols = ['MajorityChoice', 'AgeYears', 'Age_c', 'Age_c2', 'Site', 'Female', 'MajorityFirst']
    df = df[final_cols]

    # Reset index for clean downstream use
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression predicting MajorityChoice from age (centered), nonlinear age (Age_c2),
    site (categorical), gender, order, and an Age x Site interaction to test whether developmental
    trajectories differ across cultural contexts.

    Returns the fitted statsmodels binary logistic regression results object.
    """
    import statsmodels.formula.api as smf

    # Ensure the DataFrame contains the expected columns
    required = ['MajorityChoice', 'Age_c', 'Age_c2', 'Site', 'Female', 'MajorityFirst']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Formula: main effects + Age x Site interaction
    # C(Site) treats Site as categorical. Age_c2 accounts for nonlinearity. Interaction tests whether
    # the slope of age differs across sites (i.e., cultural moderation of developmental change).
    formula = 'MajorityChoice ~ Age_c + Age_c2 + C(Site) + Female + MajorityFirst + Age_c:C(Site)'

    # Fit logistic regression (binomial family). Use maximum likelihood via statsmodels' Logit wrapper
    # smf.logit uses a logit link for binary outcomes.
    model_fit = smf.logit(formula=formula, data=df).fit(disp=False)

    # Return the fitted result object (provides .summary(), .params, .predict, etc.)
    return model_fit


