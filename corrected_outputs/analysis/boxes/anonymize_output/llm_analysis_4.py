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
    Transform the original dataset into the analysis-ready dataframe. The function:
    - renames the original columns to meaningful names,
    - drops rows with missing values in core variables,
    - creates a binary MajorityChoice outcome (1 if outcome==2, else 0),
    - encodes gender as 0/1, creates DemoFirst and Site categorical columns,
    - centers age and creates a quadratic age term,
    - returns a dataframe with only the columns needed for modeling.
    """
    df = df.copy()

    # Rename original feature columns to clear names
    df = df.rename(columns={
        'feature1': 'Outcome',      # 1=unchosen option, 2=majority option, 3=minority option
        'feature2': 'GenderCode',   # 1=girl, 2=boy
        'feature3': 'Age',          # age in years
        'feature4': 'MajorityFirst',# whether majority option was demonstrated first (0/1)
        'feature5': 'SiteID'        # site id (1..8)
    })

    # Drop rows missing any core variables
    df = df.dropna(subset=['Outcome', 'GenderCode', 'Age', 'MajorityFirst', 'SiteID'])

    # Cast to appropriate dtypes
    df['Outcome'] = df['Outcome'].astype(int)
    df['GenderCode'] = df['GenderCode'].astype(int)
    df['Age'] = df['Age'].astype(float)
    df['MajorityFirst'] = df['MajorityFirst'].astype(int)
    df['SiteID'] = df['SiteID'].astype(int)

    # Keep only valid outcome values in {1,2,3}
    df = df[df['Outcome'].isin([1, 2, 3])]

    # Dependent variable: did the child choose the majority option?
    df['MajorityChoice'] = (df['Outcome'] == 2).astype(int)

    # Control: gender coded 0 = girl, 1 = boy
    # Original coding: 1 = girl, 2 = boy -> map to 0/1
    df['Gender'] = df['GenderCode'].map({1: 0, 2: 1}).astype(int)

    # Control: whether majority was demonstrated first (already 0/1 in MajorityFirst)
    df['DemoFirst'] = df['MajorityFirst'].astype(int)

    # Site as categorical string label (keeps site identity but ensures categorical handling)
    df['Site'] = 'Site_' + df['SiteID'].astype(int).astype(str)

    # Independent variable: center age and add quadratic term to capture nonlinearity
    df['Age_c'] = df['Age'] - df['Age'].mean()
    df['Age_sq'] = df['Age_c'] ** 2

    # Subset to required columns for modelling
    df = df[['Outcome', 'MajorityChoice', 'Age', 'Age_c', 'Age_sq', 'Gender', 'DemoFirst', 'Site', 'SiteID']]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression predicting the probability of choosing the majority option.

    The model specification (fixed effects) is:
      MajorityChoice ~ Age_c + Age_sq + Gender + DemoFirst + C(Site) + Age_c:C(Site)

    This includes:
    - Age_c and Age_sq to model developmental (possibly non-linear) effects of age,
    - Gender and DemoFirst as control variables,
    - Site fixed effects to adjust for baseline differences across cultural contexts,
    - An Age_c by Site interaction to allow developmental trajectories to vary by site (i.e., culture).

    The function returns a dictionary with the fitted model and marginal effects (if available).
    """
    import statsmodels.formula.api as smf

    df = df.copy()
    # Ensure Site is treated as categorical
    df['Site'] = df['Site'].astype('category')

    # Formula: main effects + site fixed effects + interaction of age (centered) by site
    formula = 'MajorityChoice ~ Age_c + Age_sq + Gender + DemoFirst + C(Site) + Age_c:C(Site)'

    # Fit logistic regression (GLM Binomial via logit link using statsmodels' Logit formula interface)
    # Using smf.logit (MLE). For many sites / interaction terms this can be large; consider mixed-effects
    # models for random slopes in future analyses.
    model_fit = smf.logit(formula=formula, data=df).fit(disp=False)

    # Try to compute average marginal effect of Age_c (if supported)
    marg_eff = None
    try:
        marg_eff = model_fit.get_margeff(at='overall', method='dydx', atexog=None)
    except Exception:
        # If marginal effects computation fails, leave as None
        marg_eff = None

    # Return model object and marginal effects object (if computed)
    return {
        'model_fit': model_fit,
        'marginal_effects': marg_eff
    }


