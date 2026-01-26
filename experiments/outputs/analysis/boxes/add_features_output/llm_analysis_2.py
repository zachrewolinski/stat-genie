from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/boxes/add_features_output/boxes.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to create analytic columns required for modeling.

    Created columns (all names used later in modeling):
    - Age_c: mean-centered age (age - mean(age)).
    - is_girl: 1 if gender == 1 (girl), 0 otherwise.
    - SocialFollow: binary indicator whether child followed any social information (majority or minority) -> y in {2,3}.
    - MajorityChoice: binary indicator whether child chose the majority option (y == 2).
    - culture: cast to categorical (keeps original numeric IDs but is categorical for modeling).

    Removes rows with missing values in any variables needed for primary analyses.
    """

    df = df.copy()

    # Required columns for our analysis
    required_cols = ['y', 'age', 'gender', 'culture', 'majority_first', 'religiousness', 'calworks']
    # Drop rows missing any of the required analytic columns
    df = df.dropna(subset=required_cols)

    # Create mean-centered age (Age_c)
    df['Age_c'] = df['age'] - df['age'].mean()

    # Binary gender indicator: 1 = girl (gender == 1), 0 = boy (gender == 2 or other)
    df['is_girl'] = (df['gender'] == 1).astype(int)

    # Cast culture to categorical for formula-based modeling
    df['culture'] = df['culture'].astype('category')

    # Derived dependent variables for targeted analyses
    # SocialFollow: followed social information (majority or minority)
    df['SocialFollow'] = df['y'].isin([2, 3]).astype(int)

    # MajorityChoice: chose the majority option (y == 2)
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Keep columns that will be used in modeling and return full frame for flexibility
    # (We keep original columns plus derived ones.)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run two complementary statistical analyses addressing the research question:
    1) Logistic regression predicting whether a child relied on social information at all (SocialFollow: chose majority or minority vs. undemonstrated option).
    2) Among children who followed social information, logistic regression predicting preference for the majority (MajorityChoice: majority vs. minority).

    Both models test the Age_c * C(culture) interaction to evaluate whether developmental change differs across cultures. Both include the same set of covariates: is_girl, majority_first, religiousness, calworks.

    Returns a dict with fitted model objects (statsmodels results).
    """

    import statsmodels.formula.api as smf
    results = {}

    # Ensure transformed columns exist
    needed = ['SocialFollow', 'MajorityChoice', 'Age_c', 'culture', 'is_girl', 'majority_first', 'religiousness', 'calworks']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Model 1: Social follow (any social information) ~ Age * Culture + covariates
    # Use interaction Age_c * C(culture) to allow age effect to vary by culture
    formula1 = 'SocialFollow ~ Age_c * C(culture) + is_girl + majority_first + religiousness + calworks'
    model1 = smf.logit(formula1, data=df).fit(disp=False)
    results['social_follow_model'] = model1

    # Model 2: Among children who followed social information, predict majority vs minority
    df_social = df[df['SocialFollow'] == 1].copy()
    if df_social.shape[0] < 20:
        # Small-sample safeguard: still attempt to fit but warn the user (returned object may be unreliable)
        import warnings
        warnings.warn('Fewer than 20 observations follow social information; majority-choice model may be unstable.')

    formula2 = 'MajorityChoice ~ Age_c * C(culture) + is_girl + majority_first + religiousness + calworks'
    model2 = smf.logit(formula2, data=df_social).fit(disp=False)
    results['majority_choice_model'] = model2

    # Optional: also fit a multinomial check (as robustness): predicted 3-category y using MNLogit
    try:
        import statsmodels.api as sm
        # Prepare exogenous matrix with dummies for culture (drop one), add constant
        exog = pd.get_dummies(df[['Age_c', 'is_girl', 'majority_first', 'religiousness', 'calworks', 'culture']], drop_first=True)
        exog = sm.add_constant(exog, has_constant='add')
        endog = df['y']
        mn_model = sm.MNLogit(endog, exog).fit(disp=False)
        results['mnlogit_full_y'] = mn_model
    except Exception:
        # If MNLogit fails (e.g., separability or dimensionality), skip gracefully
        results['mnlogit_full_y'] = None

    return results


