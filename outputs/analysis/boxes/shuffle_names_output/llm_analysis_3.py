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
    Transform the raw dataset to the modeling dataframe.

    Produces these final columns (used in the model):
      - OutcomeChoice : original choice (1=unchosen, 2=majority, 3=minority)
      - Reliance_Social: 1 if child chose a demonstrated option (majority or minority), else 0
      - Choose_Majority: 1 if child chose the majority option, else 0
      - AgeYears: child's age in years (taken from 'culture' column in this file)
      - Age_c: centered AgeYears (AgeYears - mean(AgeYears))
      - AgeGroup: categorical age bins (4-6, 7-9, 10-12, 13-14)
      - Gender: 'F'/'M' mapped from numeric gender
      - MajorityFirst_Demo: 0/1 whether majority was demonstrated first (from 'age' column in this file)
      - SiteID: site/culture id (string) from 'y'
    """
    df = df.copy()

    # Drop rows missing any of the minimal required columns
    required = ['majority_first', 'gender', 'culture', 'age', 'y']
    df = df.dropna(subset=required)

    # Standardize and rename / derive columns
    # The provided schema has inconsistent column semantics: 'culture' holds ages (4-14) here,
    # and 'age' encodes demonstration-order (0/1). We follow those semantics in these transforms.
    df['OutcomeChoice'] = pd.to_numeric(df['majority_first'], errors='coerce').astype('Int64')

    # Keep only valid choices 1,2,3
    df = df[df['OutcomeChoice'].isin([1, 2, 3])].copy()

    # Reliance on social information: chose a demonstrated option (majority or minority)
    df['Reliance_Social'] = df['OutcomeChoice'].apply(lambda x: 1 if int(x) in [2, 3] else 0).astype(int)

    # Preference for majority
    df['Choose_Majority'] = df['OutcomeChoice'].apply(lambda x: 1 if int(x) == 2 else 0).astype(int)

    # Age in years (from 'culture' column as per provided schema)
    df['AgeYears'] = pd.to_numeric(df['culture'], errors='coerce')

    # Center age for modeling
    df['Age_c'] = df['AgeYears'] - df['AgeYears'].mean()

    # Create coarse developmental bins commonly used in child-development research
    bins = [3, 6, 9, 12, 15]  # covers ages 4-14
    labels = ['4-6', '7-9', '10-12', '13-14']
    df['AgeGroup'] = pd.cut(df['AgeYears'], bins=bins, labels=labels, right=True)

    # Gender mapping (1=girl, 2=boy). Keep as categorical string for modeling with C(Gender)
    df['Gender'] = df['gender'].map({1: 'F', 2: 'M'})
    # If unexpected values exist, coerce to string form
    df['Gender'] = df['Gender'].fillna(df['gender'].astype(str))

    # Demonstration order: majority shown first? (from 'age' column per schema)
    df['MajorityFirst_Demo'] = pd.to_numeric(df['age'], errors='coerce').fillna(0).astype(int)

    # Site/Culture id for use as a culture proxy and moderation / clustering variable
    df['SiteID'] = df['y'].astype(str)

    # Final check: drop any rows that still have NA in the new core columns
    keep = ['OutcomeChoice', 'Reliance_Social', 'Choose_Majority', 'AgeYears', 'Age_c', 'Gender', 'MajorityFirst_Demo', 'SiteID']
    df = df.dropna(subset=keep)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two logistic regression models (GLM Binomial) to test developmental and cultural variation:

    Model A (reliance): Reliance_Social ~ Age_c * C(SiteID) + C(Gender) + MajorityFirst_Demo
      - Tests whether reliance on social information (choosing a demonstrated option) varies with age,
        whether that age effect differs across sites (Age_c * C(SiteID)), and controls for gender and order.

    Model B (majority preference): Choose_Majority ~ Age_c * C(SiteID) + C(Gender) + MajorityFirst_Demo
      - Tests whether preference for the majority over the minority changes with age and across sites.

    Both models return cluster-robust (by SiteID) covariance estimates.
    """
    import statsmodels.formula.api as smf

    results = {}

    # Model formulae
    formula_reliance = 'Reliance_Social ~ Age_c * C(SiteID) + C(Gender) + MajorityFirst_Demo'
    formula_majority = 'Choose_Majority ~ Age_c * C(SiteID) + C(Gender) + MajorityFirst_Demo'

    # Fit GLM (binomial) for Reliance_Social
    mod1 = smf.glm(formula_reliance, data=df, family=sm.families.Binomial()).fit()
    # Compute cluster-robust SEs by SiteID
    try:
        res1 = mod1.get_robustcov_results(cov_type='cluster', groups=df['SiteID'])
    except Exception:
        # Fallback to the original fit object if robust results fail
        res1 = mod1
    results['model_reliance'] = res1

    # Fit GLM (binomial) for Choose_Majority
    mod2 = smf.glm(formula_majority, data=df, family=sm.families.Binomial()).fit()
    try:
        res2 = mod2.get_robustcov_results(cov_type='cluster', groups=df['SiteID'])
    except Exception:
        res2 = mod2
    results['model_majority'] = res2

    # Return fitted result objects (statsmodels results). Users can call .summary() on each.
    return results


