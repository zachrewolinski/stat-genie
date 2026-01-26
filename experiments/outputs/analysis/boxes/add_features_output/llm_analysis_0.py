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
    # Work with a copy
    df = df.copy()

    # Keep only rows with the core variables present
    required_cols = ['y', 'age', 'gender', 'culture', 'majority_first', 'religiousness']
    df = df.dropna(subset=required_cols)

    # Ensure types are correct
    df['y'] = df['y'].astype(int)  # 1,2,3
    df['age'] = pd.to_numeric(df['age'], errors='coerce').astype(int)
    df['gender'] = df['gender'].astype(int)
    df['culture'] = df['culture'].astype(int)
    df['majority_first'] = df['majority_first'].astype(int)
    df['religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')

    # Standardize (z-score) age for modeling continuous effects and interactions
    df['Age_z'] = (df['age'] - df['age'].mean()) / df['age'].std(ddof=0)
    df['Age_z2'] = df['Age_z'] ** 2

    # Define coarse developmental AgeGroup for descriptive analyses and stratification
    # 4-6: early childhood, 7-9: middle childhood, 10-12: late childhood, 13-14: adolescence
    bins = [3.5, 6.5, 9.5, 12.5, 14.5]
    labels = ['Early_4_6', 'Middle_7_9', 'Late_10_12', 'Adolesc_13_14']
    df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels)

    # Derived dependent/binary outcomes for secondary analyses
    # SocialUse: did the child use social information (majority or minority) vs choose undemonstrated
    df['SocialUse'] = df['y'].isin([2, 3]).astype(int)
    # MajorityChoice: did the child choose the majority option
    df['MajorityChoice'] = (df['y'] == 2).astype(int)

    # Prepare y in 0..J-1 format for statsmodels MNLogit (original y is 1,2,3)
    df['y0'] = df['y'] - 1

    # Keep only rows that still have no missing values in newly created columns
    keep_cols = ['y', 'y0', 'age', 'Age_z', 'Age_z2', 'AgeGroup', 'culture', 'gender', 'majority_first', 'religiousness', 'SocialUse', 'MajorityChoice']
    df = df[keep_cols].dropna()

    # Ensure categorical columns are appropriately typed for downstream modeling (patsy/statsmodels)
    df['culture'] = df['culture'].astype('category')
    df['gender'] = df['gender'].astype('category')
    df['majority_first'] = df['majority_first'].astype('category')

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a multinomial logistic regression predicting choice (y: 1=undemonstrated, 2=majority, 3=minority)
    from age, culture, and controls, including Age_z x culture interactions to test whether
    developmental change differs across sites. Also fit two logistic regressions for secondary
    outcomes: SocialUse (used social info vs not) and MajorityChoice (chose majority vs not).

    Returns a dictionary with fitted statsmodels results objects.
    """
    import patsy
    import statsmodels.api as sm

    # Formula for covariates and interactions (no response variable in the RHS formula for patsy.dmatrix)
    rhs = 'Age_z + Age_z2 + C(culture) + gender + majority_first + religiousness + Age_z:C(culture)'

    # Build design matrix; patsy will add an intercept column by default
    X = patsy.dmatrix(rhs, data=df, return_type='dataframe')

    # Endog for multinomial must be integer-coded 0..(J-1)
    y = df['y0'].astype(int)

    # Multinomial logistic regression (reference category will be the first integer class, here 0 -> original y==1)
    mnlogit_model = sm.MNLogit(y, X)
    mnlogit_res = mnlogit_model.fit(method='newton', maxiter=100, disp=False)

    # Secondary binary logistic regressions using the same design matrix X
    # (1) SocialUse: used social information (majority or minority) vs chose undemonstrated
    logit_social = sm.Logit(df['SocialUse'], X)
    logit_social_res = logit_social.fit(disp=False)

    # (2) MajorityChoice: chose majority vs not
    logit_majority = sm.Logit(df['MajorityChoice'], X)
    logit_majority_res = logit_majority.fit(disp=False)

    # Return results; callers can inspect .summary() or params / conf_int() for inference
    return {
        'mnlogit_result': mnlogit_res,
        'social_logit_result': logit_social_res,
        'majority_logit_result': logit_majority_res
    }

# Example usage (not executed here):
# df_trans = transform(raw_df)
# results = model(df_trans)
# print(results['mnlogit_result'].summary())
# print(results['social_logit_result'].summary())
# print(results['majority_logit_result'].summary())


