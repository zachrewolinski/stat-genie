from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/replace_with_rvs_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair (1978) dataset into a modeling dataframe.

    Outputs (columns) included for modeling:
    - affairs_count: integer count of extramarital affairs (from 'affairs')
    - topcoded_affairs: indicator if affairs value is at or above the top-code (12)
    - HasChildren: binary indicator for presence of children (1=yes, 0=no)
    - female: binary indicator for female (1=female, 0=male)
    - age, age_c: raw age and age centered
    - yearsmarried, yearsmarried_c: raw years married and centered
    - religiousness, education, occupation, rating: kept as numeric controls
    """
    df = df.copy()

    # Ensure key columns exist
    expected_cols = ['affairs', 'children', 'gender', 'age', 'yearsmarried',
                     'religiousness', 'education', 'occupation', 'rating']

    # Convert affairs to numeric and create count variable
    df['affairs'] = pd.to_numeric(df['affairs'], errors='coerce')
    df = df.dropna(subset=['affairs', 'children', 'gender'])

    # Keep original numeric as integer count. The dataset uses special codes (7,12) but they are numeric values
    df['affairs_count'] = df['affairs'].astype(int)

    # Indicator for top-coding (12 used in this dataset for frequent affairs)
    df['topcoded_affairs'] = (df['affairs_count'] >= 12).astype(int)

    # Map children to binary 1/0. Accept common string variants.
    def map_children(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s in ('yes', 'y', '1', 'true', 't'):
            return 1
        if s in ('no', 'n', '0', 'false', 'f'):
            return 0
        return np.nan
    df['HasChildren'] = df['children'].apply(map_children)

    # Map gender to binary female indicator
    def map_female(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s.startswith('f'):
            return 1
        if s.startswith('m'):
            return 0
        return np.nan
    df['female'] = df['gender'].apply(map_female)

    # Ensure numeric controls are numeric; drop rows with missing control values used in the model
    num_cols = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with NA in any of the modeling variables
    model_cols = ['affairs_count', 'HasChildren', 'female'] + num_cols
    df = df.dropna(subset=model_cols)

    # Center continuous covariates for interpretability
    df['age_c'] = df['age'] - df['age'].mean()
    df['yearsmarried_c'] = df['yearsmarried'] - df['yearsmarried'].mean()

    # Final returned dataframe contains all columns necessary for the model
    out_cols = [
        'affairs_count', 'topcoded_affairs', 'HasChildren', 'female',
        'age', 'age_c', 'yearsmarried', 'yearsmarried_c',
        'religiousness', 'education', 'occupation', 'rating'
    ]

    return df[out_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a primary Negative Binomial regression for count outcome 'affairs_count' and
    provide an OLS robustness check. The primary specification estimates the effect
    of having children (HasChildren) on the number of extramarital affairs, controlling
    for gender, age, years married, religiousness, education, occupation, and marriage rating.

    Returns a dictionary with fitted model results objects.
    """
    import statsmodels.formula.api as smf

    # Formula: primary specification
    formula = ('affairs_count ~ HasChildren + female + age_c + yearsmarried_c + '
               'religiousness + education + occupation + rating')

    # 1) Negative Binomial (recommended for over-dispersed count data)
    try:
        nb_res = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial()).fit()
    except Exception:
        # Fallback: statsmodels discrete NegativeBinomial if GLM family fails
        from statsmodels.discrete.count_model import NegativeBinomial
        nb_mod = NegativeBinomial(df['affairs_count'], sm.add_constant(df[['HasChildren','female','age_c','yearsmarried_c','religiousness','education','occupation','rating']]))
        nb_res = nb_mod.fit(disp=False)

    # 2) OLS robustness check (linear model on counts)
    ols_res = smf.ols(formula=formula, data=df).fit()

    # Compute a simple overdispersion statistic (variance/mean) for affairs_count
    mean_count = df['affairs_count'].mean()
    var_count = df['affairs_count'].var()
    overdispersion = var_count / mean_count if mean_count > 0 else np.nan

    results = {
        'negative_binomial': nb_res,
        'ols': ols_res,
        'overdispersion_stat': overdispersion,
        'n_obs': int(df.shape[0])
    }

    return results


