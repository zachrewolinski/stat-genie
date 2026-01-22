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
    Transform raw dataset to analysis-ready dataframe with the exact column names used in the model.

    Inputs expected (original columns):
      - feature1: outcome (1=unchosen, 2=majority, 3=minority)
      - feature2: gender (1=girl, 2=boy)
      - feature3: age in years (4-14)
      - feature4: whether majority was demonstrated first (0/1)
      - feature5: site id (1..8)

    Outputs (added / cleaned columns):
      - Choice: categorical string label ('unchosen','majority','minority')
      - ChoiceNum: integer 0/1/2 mapping for multinomial modeling (0=unchosen,1=majority,2=minority)
      - DemonstratedChosen: binary (1 if majority or minority chosen, else 0)
      - AgeYears: same as feature3 (float)
      - Male: 1 if boy, 0 if girl
      - SiteID: categorical site label like 'Site_1'
      - OrderMajorityFirst: same as feature4 (0/1)
      - DevelopmentStage: categorical age bins '4-6','7-9','10-12','13-14'
    """
    df = df.copy()

    # Keep only rows with non-missing critical fields
    df = df.dropna(subset=['feature1', 'feature3', 'feature5'])

    # Map choices to readable labels and numeric codes
    choice_map = {1: 'unchosen', 2: 'majority', 3: 'minority'}
    df['Choice'] = df['feature1'].map(choice_map).astype('category')
    # Numeric mapping for MNLogit: 0,1,2
    num_map = {1: 0, 2: 1, 3: 2}
    df['ChoiceNum'] = df['feature1'].map(num_map).astype(int)

    # Binary: did the child choose one of the demonstrated options (majority or minority)
    df['DemonstratedChosen'] = df['feature1'].isin([2, 3]).astype(int)

    # Age
    df['AgeYears'] = pd.to_numeric(df['feature3'], errors='coerce')

    # Gender -> Male (1=boy, 0=girl). If missing or other, set to NaN
    df['Male'] = df['feature2'].map({1: 0, 2: 1})

    # Site as categorical string (useable in formulas)
    df['SiteID'] = 'Site_' + df['feature5'].astype(int).astype(str)
    df['SiteID'] = df['SiteID'].astype('category')

    # Order variable (ensure numeric 0/1)
    df['OrderMajorityFirst'] = pd.to_numeric(df['feature4'], errors='coerce').fillna(0).astype(int)

    # Developmental stage bins - labels chosen to reflect age ranges in dataset
    bins = [3.999, 6.0, 9.0, 12.0, 14.1]
    labels = ['4-6', '7-9', '10-12', '13-14']
    df['DevelopmentStage'] = pd.cut(df['AgeYears'], bins=bins, labels=labels, include_lowest=True).astype('category')

    # Optionally drop rows with missing derived key covariates (Age or Gender missing)
    # we keep rows where AgeYears is present; gender may be used as control but we won't drop rows missing Male here
    df = df.dropna(subset=['AgeYears'])

    # Return only the columns necessary for modeling (plus originals if desired)
    # Keep originals plus derived columns for traceability
    keep_cols = list(df.columns)
    # But ensure the final dataframe contains the required model columns
    required_cols = [
        'Choice', 'ChoiceNum', 'DemonstratedChosen',
        'AgeYears', 'Male', 'SiteID', 'OrderMajorityFirst', 'DevelopmentStage'
    ]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Required column missing after transform: {c}")

    return df[required_cols + [c for c in keep_cols if c not in required_cols]]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two models to answer the research question:
      1) A multinomial logistic regression predicting Choice (unchosen / majority / minority) using AgeYears, SiteID, Male, OrderMajorityFirst and Age x Site interactions. This tests whether developmental trajectories (age effects) on choice differ across cultures.
      2) A binary logistic regression predicting DemonstratedChosen (0/1) with AgeYears * C(SiteID) to test whether general reliance on social information (any demonstrated option) changes with age differently across sites.

    Returns a dict with fitted model results objects.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    # Work on a copy
    d = df.copy()

    # --- Multinomial model setup (MNLogit expects numeric endogenous) ---
    # Endog: ChoiceNum (0=unchosen,1=majority,2=minority)
    endog = d['ChoiceNum'].astype(int)

    # Build exog with site dummy variables (drop first to avoid multicollinearity)
    site_dummies = pd.get_dummies(d['SiteID'], prefix='Site', drop_first=True)

    # Basic covariates
    base_covs = d[['AgeYears', 'Male', 'OrderMajorityFirst']].copy()

    exog = pd.concat([base_covs, site_dummies], axis=1)

    # Add Age x Site interaction terms to test whether age effects vary by site
    for col in site_dummies.columns:
        exog[f'Age_x_{col}'] = exog['AgeYears'] * exog[col]

    exog = sm.add_constant(exog, has_constant='add')

    # Fit multinomial logistic regression
    try:
        mnlogit_mod = sm.MNLogit(endog, exog)
        mnlogit_res = mnlogit_mod.fit(method='newton', maxiter=100, disp=False)
    except Exception as e:
        # If MNLogit fails to converge, try alternative solver or return the exception
        mnlogit_res = {'error': str(e)}

    # --- Binary logistic regression for DemonstratedChosen ---
    # Use a formula with C(SiteID) so statsmodels creates dummies automatically and we can include AgeYears * C(SiteID)
    formula = 'DemonstratedChosen ~ AgeYears * C(SiteID) + Male + OrderMajorityFirst'
    try:
        logit_mod = smf.logit(formula, data=d)
        logit_res = logit_mod.fit(disp=False)
    except Exception as e:
        logit_res = {'error': str(e)}

    # Return both fitted results (or error messages) so downstream code can inspect summaries / coefficients
    return {
        'mnlogit_result': mnlogit_res,
        'logit_result': logit_res
    }


