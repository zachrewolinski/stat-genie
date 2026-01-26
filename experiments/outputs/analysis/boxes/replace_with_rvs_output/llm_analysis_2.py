from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/replace_with_rvs_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modeling.

    Produces the following new/renamed columns used in the analyses:
    - OutcomeY: integer copy of original y (1=unchosen, 2=majority, 3=minority)
    - Age: copy of age as float
    - Age_c: age centered around sample mean
    - Age2: squared centered age (to capture quadratic effects)
    - gender_boy: 1 if original gender==2 (boy), 0 if gender==1 (girl)
    - majority_first: integer copy of original majority_first (0/1)
    - culture: integer copy of original culture (will be used as categorical in models)
    - SocialCopy: binary 1 if child copied any demonstrator (majority or minority), 0 if chose undemonstrated
    - MajorityChoice: binary 1 if child chose majority (y==2), 0 otherwise
    """
    df = df.copy()

    # Drop rows with missing critical fields
    df = df.dropna(subset=['y', 'age', 'culture', 'gender', 'majority_first'])

    # Standardize / cast columns
    df['OutcomeY'] = df['y'].astype(int)
    df['Age'] = df['age'].astype(float)
    df['culture'] = df['culture'].astype(int)
    # majority_first in data is 0/1; ensure integer type
    df['majority_first'] = df['majority_first'].astype(int)

    # Center age and add quadratic term (centered squared)
    df['Age_c'] = df['Age'] - df['Age'].mean()
    df['Age2'] = df['Age_c'] ** 2

    # Gender binary: create gender_boy (1=boy when gender==2, 0=girl when gender==1)
    df['gender_boy'] = df['gender'].apply(lambda x: 1 if int(x) == 2 else 0).astype(int)

    # Behavioral derived binaries
    df['SocialCopy'] = df['OutcomeY'].apply(lambda x: 1 if int(x) in (2, 3) else 0).astype(int)
    df['MajorityChoice'] = df['OutcomeY'].apply(lambda x: 1 if int(x) == 2 else 0).astype(int)

    # Keep only columns necessary for downstream analyses (retain originals for traceability)
    # We'll return the full dataframe but ensure the necessary columns exist
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two complementary models to answer the research question:
      1) SocialCopy model (binary logistic): Do children copy demonstrators (majority or minority) vs. pick the undemonstrated option? Predictors: age (linear + quadratic), culture (categorical), gender_boy, majority_first, and Age x Culture interactions.
      2) MajorityChoice model (binary logistic, fit only among children who copied someone): Among copiers, do children preferentially choose the majority (vs. minority)? Same predictors.

    Returns a dict with keys 'socialcopy_model' and 'majority_model' containing fitted statsmodels results objects. If there are too few copiers to fit the majority model, returns None for that entry.
    """
    import statsmodels.formula.api as smf

    # Work on a copy
    df = df.copy()

    # Ensure required columns exist
    required = ['SocialCopy', 'MajorityChoice', 'Age_c', 'Age2', 'culture', 'gender_boy', 'majority_first']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Model 1: SocialCopy ~ Age_c + Age2 + C(culture) + gender_boy + majority_first + Age_c:C(culture)
    # Use categorical coding for culture via C(culture). This fits a separate intercept for each culture and allows interaction terms.
    formula_sc = 'SocialCopy ~ Age_c + Age2 + C(culture) + gender_boy + majority_first + Age_c:C(culture)'
    socialcopy_model = smf.logit(formula=formula_sc, data=df).fit(disp=False)

    # Model 2: MajorityChoice among those who copied (SocialCopy==1)
    df_copiers = df[df['SocialCopy'] == 1].copy()
    if df_copiers.shape[0] < 30:
        # If sample is too small, we still attempt but warn the user; here we return None to indicate limited data for reliable inference
        majority_model = None
    else:
        formula_mc = 'MajorityChoice ~ Age_c + Age2 + C(culture) + gender_boy + majority_first + Age_c:C(culture)'
        majority_model = smf.logit(formula=formula_mc, data=df_copiers).fit(disp=False)

    # Return fitted result objects for downstream inspection (coefficients, p-values, CIs, predicted probabilities, etc.)
    return {
        'socialcopy_model': socialcopy_model,
        'majority_model': majority_model
    }


