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
    # Work on a copy
    df = df.copy()

    # Required source columns (based on provided schema):
    # - 'majority_first': outcome encoding 1=unchosen, 2=majority, 3=minority
    # - 'culture': contains the child's age in years (per schema samples)
    # - 'y': site id (1..8)
    # - 'gender': 1=girl, 2=boy
    # - 'age': (per schema) indicator whether majority option was demonstrated first (0/1)

    # Drop rows missing any of the required variables
    df = df.dropna(subset=['majority_first', 'culture', 'y', 'gender', 'age'])

    # Dependent variable: binary indicator for choosing the majority option
    # According to the schema: majority is encoded as 2
    df['ChoiceMajority'] = (df['majority_first'] == 2).astype(int)

    # Age: in the provided schema 'culture' appears to contain ages (4-14)
    # Convert to numeric and store as AgeYears
    df['AgeYears'] = pd.to_numeric(df['culture'], errors='coerce')

    # Site / cultural context: create a categorical label from site id column 'y'
    df['Site'] = 'Site_' + df['y'].astype(int).astype(str)
    df['Site'] = df['Site'].astype('category')

    # Gender: map to a male indicator (1 = boy, 0 = girl)
    df['GenderMale'] = (df['gender'] == 2).astype(int)

    # MajorityFirst: indicator whether majority was demonstrated first (source column 'age' per schema)
    df['MajorityFirst'] = df['age'].astype(int)

    # Center age and add quadratic term to allow non-linear developmental effects
    df['Age_c'] = df['AgeYears'] - df['AgeYears'].mean()
    df['Age_c2'] = df['Age_c'] ** 2

    # Keep only the columns needed for modeling (and in the conceptual variables)
    result_cols = ['ChoiceMajority', 'AgeYears', 'Age_c', 'Age_c2', 'Site', 'GenderMale', 'MajorityFirst']
    df_out = df[result_cols].copy()

    return df_out


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """Runs a binomial (logistic) regression predicting choice of majority option.
    Model specification:
      ChoiceMajority ~ Age_c + Age_c2 + C(Site) + GenderMale + MajorityFirst + Age_c:C(Site)
    - C(Site) includes site fixed effects to control for average differences across cultures.
    - Age_c:C(Site) tests whether the age-developmental slope differs across sites (culture x age interaction).
    - GenderMale and MajorityFirst are included as nuisance covariates.

    Returns the fitted GLM (statsmodels) results object.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure Site is categorical
    df = df.copy()
    df['Site'] = df['Site'].astype('category')

    # Formula with interaction between age (centered) and site (culture)
    formula = 'ChoiceMajority ~ Age_c + Age_c2 + C(Site) + GenderMale + MajorityFirst + Age_c:C(Site)'

    # Fit a GLM with binomial family (logistic regression)
    model_glm = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Print a summary for quick inspection (can be removed if not desired)
    print(model_glm.summary())

    return model_glm


