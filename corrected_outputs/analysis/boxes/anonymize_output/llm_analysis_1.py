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
    Transform the raw dataset into a dataframe with columns used in modeling.

    Input schema:
      - feature1: Outcome (1=unchosen option, 2=majority option, 3=minority option)
      - feature2: Gender (1=girl, 2=boy)
      - feature3: Age (years)
      - feature4: whether majority was demonstrated first (0/1)
      - feature5: Site ID (1..8)

    Output columns required by the model:
      - IsMajority: binary (1 if feature1==2 else 0)
      - Age: numeric age in years
      - Age_c: centered age (Age - mean(Age))
      - Age_z: standardized age (z-score)
      - Gender: categorical ('Female'/'Male')
      - MajorityFirst: integer 0/1 from feature4
      - Site: categorical site identifier string 'Site_<id>'
    """
    df = df.copy()

    # Drop rows missing critical fields
    required = ['feature1', 'feature3', 'feature5']
    df = df.dropna(subset=required)

    # Dependent variable: chose majority (feature1 == 2)
    df['IsMajority'] = (df['feature1'] == 2).astype(int)

    # Age variable and transformations
    df['Age'] = pd.to_numeric(df['feature3'], errors='coerce')
    df = df.dropna(subset=['Age'])
    df['Age_c'] = df['Age'] - df['Age'].mean()
    df['Age_z'] = (df['Age'] - df['Age'].mean()) / (df['Age'].std(ddof=0) if df['Age'].std(ddof=0) != 0 else 1.0)

    # Gender: map to readable categories; keep as categorical
    df['Gender'] = df['feature2'].map({1: 'Female', 2: 'Male'})
    # If mapping creates NaNs (unexpected codes), keep original value as string
    df['Gender'] = df['Gender'].fillna(df['feature2'].astype(str)).astype('category')

    # MajorityFirst: ensure binary 0/1
    df['MajorityFirst'] = df['feature4'].fillna(0).astype(int)

    # Site: create categorical site labels
    df['Site'] = 'Site_' + df['feature5'].astype(int).astype(str)
    df['Site'] = df['Site'].astype('category')

    # Keep only columns necessary for modeling plus Age and Age_z for diagnostics
    required_cols = ['IsMajority', 'Age', 'Age_c', 'Age_z', 'Gender', 'MajorityFirst', 'Site']
    df = df[required_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a logistic regression predicting the probability of choosing the majority option.

    Model specification:
      - Outcome: IsMajority (binary)
      - Predictors: Age_c (centered age), site (categorical, C(Site)), their interaction Age_c * C(Site)
      - Controls: Gender (categorical), MajorityFirst (0/1)

    We fit a binomial (logit) generalized linear model with site fixed effects and Age-by-Site interactions
    to estimate how developmental slopes differ across cultural contexts. We return cluster-robust
    standard-error results clustered by Site.
    """
    import statsmodels.formula.api as smf

    # Ensure categorical coding in dataframe matches formula expectations
    df = df.copy()
    df['Site'] = df['Site'].astype('category')
    df['Gender'] = df['Gender'].astype('category')

    # Formula: main effect of age, site fixed effects, their interaction, plus controls
    formula = 'IsMajority ~ Age_c * C(Site) + C(Gender) + MajorityFirst'

    # Fit logistic regression
    model_fit = smf.logit(formula=formula, data=df).fit(disp=False)

    # Compute cluster-robust (by Site) covariance estimator
    try:
        results = model_fit.get_robustcov_results(cov_type='cluster', groups=df['Site'])
    except Exception:
        # Fallback to default results if clustering fails
        results = model_fit

    return results


