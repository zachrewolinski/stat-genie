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
    # Work on a copy
    df = df.copy()

    # Drop rows missing essential variables for this analysis
    df = df.dropna(subset=['feature1', 'feature3', 'feature5', 'feature2', 'feature4'])

    # Dependent variable: did the child choose the majority option? (feature1: 2 = majority)
    df['MajorityChoice'] = (df['feature1'] == 2).astype(int)

    # Independent variable: Age (years)
    df['Age'] = pd.to_numeric(df['feature3'], errors='coerce')

    # Control: gender -> IsFemale (1 = girl, 0 = boy). Map unexpected values to NaN then drop earlier.
    df['IsFemale'] = df['feature2'].map({1: 1, 2: 0}).astype(float)

    # Control: whether majority was demonstrated first (feature4 already 0/1)
    df['MajorityFirst'] = df['feature4'].astype(int)

    # Site / culture identifier (treat as categorical in model)
    # Keep as integer category column used with C(SiteID) in modeling
    df['SiteID'] = df['feature5'].astype(int)

    # Center age to improve interpretability and numerical stability
    df['Age_c'] = df['Age'] - df['Age'].mean()
    df['Age_c_sq'] = df['Age_c'] ** 2

    # Final safety: drop rows with any remaining NA in columns used by the model
    model_cols = ['MajorityChoice', 'Age', 'Age_c', 'Age_c_sq', 'SiteID', 'IsFemale', 'MajorityFirst']
    df = df.dropna(subset=model_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    import statsmodels.formula.api as smf
    # Work on a copy
    df = df.copy()

    # Formula: logistic regression predicting likelihood of choosing the majority
    # Include Age (centered), quadratic age term, Site (categorical) and Age x Site interactions
    # Controls: child's gender and whether majority was shown first
    formula = 'MajorityChoice ~ Age_c * C(SiteID) + Age_c_sq + IsFemale + MajorityFirst'

    # Fit a binomial (logistic) GLM
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Return the fitted results object (user can inspect summary, params, conf_int, predict, etc.)
    return model


