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
    Transform the raw dataset into a cleaned dataframe containing the columns required for modeling.

    Input expected columns in the raw dataframe:
      - 'majority_first': original categorical choice outcome (1 = unchosen option, 2 = majority option, 3 = minority option)
      - 'gender': 1 = girl, 2 = boy
      - 'culture': numeric age in years (note: in the provided schema this column holds the child's age)
      - 'age': binary flag indicating whether the majority option was demonstrated first (0/1)
      - 'y': site id (1..8)

    Output columns (kept/created):
      - 'Choice' (int): original 1/2/3 outcome copied from 'majority_first'
      - 'ChoseMajority' (int): 1 if Choice == 2 (majority), else 0
      - 'AgeYears' (float): child's age in years (from 'culture')
      - 'AgeGroup' (category): coarse developmental bins (4-6, 7-9, 10-14)
      - 'Site' (str category): site id as string (from 'y')
      - 'IsMale' (int): 1 if gender == 2, else 0
      - 'MajorityFirst' (int): copy of 'age' column indicating whether majority option was demonstrated first
    """
    df = df.copy()

    # Drop rows missing any of the columns needed for analysis
    need_cols = ['majority_first', 'gender', 'culture', 'age', 'y']
    df = df.dropna(subset=need_cols)

    # Standardize / rename columns
    # Original schema has some naming mismatch: 'culture' actually contains age in years; 'age' contains majority-first flag.
    df['majority_first'] = pd.to_numeric(df['majority_first'], errors='coerce').astype('Int64')
    df['Choice'] = df['majority_first'].astype(int)

    # Binary DV: chose majority (choice == 2)
    df['ChoseMajority'] = (df['Choice'] == 2).astype(int)

    # Age in years
    df['AgeYears'] = pd.to_numeric(df['culture'], errors='coerce')

    # Keep rows with plausible ages (dataset describes ages 4-14)
    df = df[df['AgeYears'].between(4, 14)]

    # Create coarse age groups for descriptive checks / stratified analyses
    bins = [0, 6, 9, 14]
    labels = ['4-6', '7-9', '10-14']
    df['AgeGroup'] = pd.cut(df['AgeYears'], bins=bins, labels=labels, include_lowest=True)

    # Site as categorical string (use original 'y' column which is site id)
    df['Site'] = df['y'].astype(int).astype(str)

    # Gender -> IsMale (1 = boy, 0 = girl)
    df['IsMale'] = (pd.to_numeric(df['gender'], errors='coerce') == 2).astype(int)

    # MajorityFirst flag (original 'age' column per schema is whether majority was shown first)
    df['MajorityFirst'] = pd.to_numeric(df['age'], errors='coerce').astype(int)

    # Keep only the columns required for modeling and reset index
    out_cols = ['Choice', 'ChoseMajority', 'AgeYears', 'AgeGroup', 'Site', 'IsMale', 'MajorityFirst']
    df = df.loc[:, out_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit two complementary models to answer whether reliance on the majority varies by age and culture:
      1) Primary analysis: logistic regression predicting the binary outcome ChoseMajority (1 = majority chosen) with fixed effects for age, gender, majority-first, site (dummy-coded) and interactions between age and site (tests whether developmental trajectories differ across sites).
      2) Secondary analysis: multinomial logistic regression predicting the three-way choice outcome (unchosen / majority / minority) with main effects for age, gender, majority-first and site (no interaction, for stability).

    Returns a dictionary with the fitted model objects (statsmodels results).
    """
    import statsmodels.api as sm

    df = df.copy()

    # Prepare site dummy variables (fixed effects). Drop first to avoid multicollinearity.
    site_dummies = pd.get_dummies(df['Site'], prefix='Site', drop_first=True)

    # Base predictors
    X_base = pd.concat([df[['AgeYears', 'IsMale', 'MajorityFirst']].reset_index(drop=True), site_dummies.reset_index(drop=True)], axis=1)

    # Build interaction terms between AgeYears and each site dummy to test whether age effects differ across sites
    X = X_base.copy()
    for col in site_dummies.columns:
        X[f'{col}:AgeYears'] = X[col] * df['AgeYears']

    # Add intercept
    X = sm.add_constant(X)

    # Binary outcome: ChoseMajority
    y_bin = df['ChoseMajority']

    # Fit logistic regression for binary outcome. Use a regularized fallback if needed.
    try:
        logit_model = sm.Logit(y_bin, X).fit(disp=False)
    except Exception as e:
        # If perfect separation / convergence issues occur, try a small L1/L2 regularization via fit_regularized
        logit_model = sm.Logit(y_bin, X).fit_regularized(disp=False)

    # Secondary: multinomial logistic regression on the full choice (1=unchosen,2=majority,3=minority)
    X_mn = sm.add_constant(X_base)
    y_mn = df['Choice'].astype(int)

    # Fit multinomial logit (no interaction to keep model stable). The baseline is the first numeric category (1 = unchosen option).
    mn_model = sm.MNLogit(y_mn, X_mn).fit(disp=False)

    results = {
        'binary_majority_logit': logit_model,
        'multinomial_choice_mnlogit': mn_model
    }

    return results


