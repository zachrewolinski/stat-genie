from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/affairs/add_features_output/affairs.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Fair (affairs) dataset into a dataframe suitable for modeling.

    Produces the following new/clean columns used in modeling:
      - children_yes: binary indicator (1=yes, 0=no)
      - gender_female: binary indicator (1=female, 0=male)
      - AnyAffair: binary indicator (1 if affairs > 0, else 0)
      - age_centered: age minus sample mean
      - yearsmarried_centered: yearsmarried minus sample mean
      - retains original 'affairs' count column for count models

    Rows with missing values in columns required for the primary analyses are dropped.
    """
    # Make a copy to avoid side effects
    df = df.copy()

    # Standardize column names (if necessary) - assume they are as described in the schema
    # Map children to binary
    if 'children' in df.columns:
        df['children_yes'] = df['children'].map({
            'yes': 1,
            'no': 0,
        })
    else:
        # if dataset already numeric 0/1 under another name, try to preserve it
        df['children_yes'] = df.get('children_yes', np.nan)

    # Map gender to binary female indicator
    if 'gender' in df.columns:
        # Accept common variations in case
        df['gender_female'] = df['gender'].astype(str).str.lower().map(lambda x: 1 if x == 'female' else (0 if x == 'male' else np.nan))
    else:
        df['gender_female'] = df.get('gender_female', np.nan)

    # Binary outcome: any affair
    df['AnyAffair'] = (df['affairs'] > 0).astype(int)

    # Center continuous covariates to aid interpretation
    for col in ['age', 'yearsmarried']:
        if col in df.columns:
            mean_val = df[col].mean()
            df[col + '_centered'] = df[col] - mean_val
        else:
            df[col + '_centered'] = np.nan

    # Ensure control numeric columns exist; if not, create with NaN to keep column list stable
    for col in ['religiousness', 'education', 'occupation', 'rating']:
        if col not in df.columns:
            df[col] = np.nan

    # Select only the columns needed for modeling (keeps original 'affairs' count as well)
    model_cols = [
        'affairs',
        'AnyAffair',
        'children_yes',
        'gender_female',
        'age_centered',
        'yearsmarried_centered',
        'religiousness',
        'education',
        'occupation',
        'rating'
    ]

    # Drop rows with missing values in core variables used by the primary models
    df = df.dropna(subset=['affairs', 'children_yes', 'gender_female', 'age_centered', 'yearsmarried_centered', 'religiousness', 'education', 'occupation', 'rating'])

    # Keep only needed columns (but preserve index)
    df = df[model_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run primary statistical models to answer: Does having children decrease engagement in extramarital affairs?

    Approaches provided:
      1) Logistic regression on AnyAffair (probability of any extramarital affair).
      2) Negative binomial regression on the positive-count subsample (affairs > 0) to examine frequency conditional on any affair.

    The logistic model is the primary model for the research question (whether children reduce the likelihood of any affair).
    We include an interaction between children_yes and gender_female to test whether the association differs by gender.

    Returns a dict with fitted model result objects.
    """
    import statsmodels.api as _sm
    import statsmodels.formula.api as smf

    results = {}

    # Formula for both models: main effect of children, interaction with gender, and controls
    formula = (
        'AnyAffair ~ children_yes * gender_female + age_centered + yearsmarried_centered '
        '+ religiousness + education + occupation + rating'
    )

    # 1) Logistic regression for probability of any affair
    try:
        logit_res = smf.logit(formula, data=df).fit(disp=False)
        results['logit_any_affair'] = logit_res
    except Exception as e:
        results['logit_any_affair'] = None
        results['logit_error'] = str(e)

    # 2) Negative binomial regression on positive counts (affairs > 0)
    df_pos = df[df['affairs'] > 0].copy()
    if df_pos.shape[0] >= 30:
        # Use count model only if we have a reasonable number of positive observations
        nb_formula = (
            'affairs ~ children_yes * gender_female + age_centered + yearsmarried_centered '
            '+ religiousness + education + occupation + rating'
        )
        try:
            nb_res = smf.glm(nb_formula, data=df_pos, family=_sm.families.NegativeBinomial()).fit()
            results['negbin_pos_affairs'] = nb_res
        except Exception as e:
            results['negbin_pos_affairs'] = None
            results['negbin_error'] = str(e)
    else:
        results['negbin_pos_affairs'] = None
        results['negbin_error'] = 'Not enough positive-affair observations for reliable count model (n={}).'.format(df_pos.shape[0])

    # Attach metadata about sample sizes
    results['n_total'] = df.shape[0]
    results['n_positive_affairs'] = int((df['affairs'] > 0).sum())
    results['n_children'] = int(df['children_yes'].sum())

    return results


