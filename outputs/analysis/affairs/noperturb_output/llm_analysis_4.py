from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/noperturb_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair (1978) dataset into a dataframe prepared for modeling.

    Produces the following new/clean columns used by the model:
    - children_yes: binary 1=yes, 0=no
    - gender_male: binary 1=male, 0=female
    - affairs: ensured numeric (dependent variable)
    - age, yearsmarried, religiousness, education, occupation, rating: ensured numeric

    Drops rows with missing values on any of the variables used in the analysis.
    Also creates an indicator for top-coded affairs (affairs_topcoded) for diagnostics.
    """
    # Work on a copy
    df = df.copy()

    # Columns required for the analysis
    required_cols = [
        'affairs', 'children', 'gender', 'age', 'yearsmarried',
        'religiousness', 'education', 'occupation', 'rating'
    ]

    # Drop rows missing any required columns
    df = df.dropna(subset=required_cols)

    # Ensure numeric types where appropriate
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['yearsmarried'] = pd.to_numeric(df['yearsmarried'], errors='coerce')
    df['religiousness'] = pd.to_numeric(df['religiousness'], errors='coerce')
    df['education'] = pd.to_numeric(df['education'], errors='coerce')
    df['occupation'] = pd.to_numeric(df['occupation'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Map children to binary indicator: 'yes' -> 1, 'no' -> 0
    df['children_yes'] = df['children'].map({
        'yes': 1,
        'no': 0
    })

    # Map gender to binary male indicator: 'male' -> 1, 'female' -> 0
    # If gender uses other labels, convert with .str.lower() and map
    df['gender'] = df['gender'].astype(str).str.lower()
    df['gender_male'] = df['gender'].map({
        'male': 1,
        'female': 0
    })

    # If mapping introduced NaNs (unexpected categories), drop them
    df = df.dropna(subset=['children_yes', 'gender_male', 'affairs', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating'])

    # Create a diagnostic flag for top-coded affairs (useful for descriptive checks / potential Tobit considerations)
    df['affairs_topcoded'] = (df['affairs'] >= 12).astype(int)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit multiple models to estimate the association between having children and reported extramarital affairs.

    Models fitted:
    1) OLS with heteroskedasticity-robust (HC3) standard errors for a baseline linear estimate.
    2) Negative Binomial GLM (counts) to account for count nature and overdispersion.
    3) Zero-Inflated Negative Binomial (ZINB) to account for excess zeros if present.

    Returns a dict with fitted results objects and key diagnostics.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

    results = {}

    # Formula with interaction between children and gender to test moderation
    formula = ('affairs ~ children_yes + gender_male + children_yes:gender_male '
               '+ age + yearsmarried + religiousness + education + occupation + rating')

    # 1) OLS (baseline) with robust SEs
    ols_model = smf.ols(formula, data=df)
    ols_res = ols_model.fit(cov_type='HC3')
    results['ols'] = ols_res

    # 2) Negative Binomial (GLM). Use a small constant to ensure positive predicted mean if necessary
    try:
        nb_model = smf.glm(formula, data=df, family=sm.families.NegativeBinomial())
        nb_res = nb_model.fit()
        results['neg_bin'] = nb_res
    except Exception as e:
        results['neg_bin_error'] = str(e)

    # 3) Zero-Inflated Negative Binomial (ZINB) - use the same exog for inflation part (can be adjusted)
    # Prepare exog and exog_infl (add constant)
    exog_vars = ['children_yes', 'gender_male', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    exog = sm.add_constant(df[exog_vars])
    exog_infl = sm.add_constant(df[['children_yes', 'gender_male']])  # simpler inflation model

    try:
        zinb = ZeroInflatedNegativeBinomialP(endog=df['affairs'], exog=exog, exog_infl=exog_infl, inflation='logit')
        zinb_res = zinb.fit(method='newton', maxiter=100, disp=False)
        results['zinb'] = zinb_res
    except Exception as e:
        # If ZINB fails to converge or not available, record the error
        results['zinb_error'] = str(e)

    # Add a small descriptive table: mean affairs by children
    desc = df.groupby('children_yes')['affairs'].agg(['count', 'mean', 'median', 'std']).to_dict()
    results['descriptive_by_children'] = desc

    return results


