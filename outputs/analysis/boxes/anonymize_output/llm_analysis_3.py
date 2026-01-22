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
    Transform raw dataset into analysis-ready dataframe.

    Input columns expected:
      - feature1: outcome 1=unchosen option, 2=majority option, 3=minority option
      - feature2: gender (1=girl, 2=boy)
      - feature3: age in years (4-14)
      - feature4: whether majority was demonstrated first (0/1)
      - feature5: site ID (integer)

    Adds the following columns used in modeling:
      - MajorityChoice: binary indicator (1 if chose majority option, else 0)
      - SocialChoice: binary indicator (1 if chose any demonstrated option (majority or minority), else 0)
      - Age: numeric age in years
      - Age_c: age mean-centered
      - AgeGroup: categorical developmental stage bins (4-6, 7-9, 10-12, 13-14)
      - Site: categorical site variable (from feature5)
      - IsMale: binary gender indicator (1 = male, 0 = female)
      - MajorityFirst: binary indicator from feature4

    Rows with missing values for any of the required variables are dropped.
    """
    df = df.copy()

    # Keep only rows with the required columns (drop rows with missing key variables)
    required = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    df = df.dropna(subset=required)

    # Ensure correct dtypes
    df['feature1'] = df['feature1'].astype(int)
    df['feature2'] = df['feature2'].astype(int)
    df['feature3'] = df['feature3'].astype(float)
    df['feature4'] = df['feature4'].astype(int)
    df['feature5'] = df['feature5'].astype(int)

    # Original choice retained for reference
    df['Choice'] = df['feature1']

    # Dependent variables
    df['MajorityChoice'] = (df['feature1'] == 2).astype(int)
    # Reliance on social information: chose any demonstrated option (majority or minority)
    df['SocialChoice'] = df['feature1'].isin([2, 3]).astype(int)

    # Age and centered age
    df['Age'] = df['feature3']
    df['Age_c'] = df['Age'] - df['Age'].mean()

    # Developmental bins (use right-inclusive bins). Bins chosen to roughly match typical developmental stages.
    df['AgeGroup'] = pd.cut(df['Age'], bins=[3, 6, 9, 12, 14], labels=['4-6', '7-9', '10-12', '13-14'], right=True)

    # Site (culture). Keep as categorical for modeling.
    df['Site'] = df['feature5'].astype('category')

    # Gender: map to binary indicator for modeling (1 = male, 0 = female)
    # According to schema: feature2: 1 = girl, 2 = boy
    df['IsMale'] = (df['feature2'] == 2).astype(int)

    # Whether majority was demonstrated first (feature4 is already 0/1 per schema)
    df['MajorityFirst'] = df['feature4'].astype(int)

    # Sanity: drop any rows with NA produced by pd.cut (should be none if ages in expected range)
    df = df.dropna(subset=['Age', 'Age_c', 'MajorityChoice', 'Site', 'IsMale', 'MajorityFirst'])

    # Final dataframe ready for modeling
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit logistic regression models to test whether (1) preference for the majority and (2) reliance on social information
    vary across age (developmental stage) and culture (site).

    Models run:
      - majority_model: MajorityChoice ~ Age_c * C(Site) + IsMale + MajorityFirst
      - social_model: SocialChoice ~ Age_c * C(Site) + IsMale + MajorityFirst

    The interaction Age_c * C(Site) tests whether age-related change (developmental slope) differs across sites.

    Returns a dict with the fitted statsmodels results objects for both models.
    """
    import statsmodels.formula.api as smf

    results = {}

    # Model 1: preference for majority option
    formula_maj = 'MajorityChoice ~ Age_c * C(Site) + IsMale + MajorityFirst'
    try:
        maj_model = smf.logit(formula_maj, data=df).fit(disp=False)
        results['majority_model'] = maj_model
    except Exception as e:
        # If the logistic model fails (e.g., perfect separation), return the exception message
        results['majority_model'] = e

    # Model 2: reliance on any social information (majority or minority demonstration)
    formula_soc = 'SocialChoice ~ Age_c * C(Site) + IsMale + MajorityFirst'
    try:
        soc_model = smf.logit(formula_soc, data=df).fit(disp=False)
        results['social_model'] = soc_model
    except Exception as e:
        results['social_model'] = e

    return results


