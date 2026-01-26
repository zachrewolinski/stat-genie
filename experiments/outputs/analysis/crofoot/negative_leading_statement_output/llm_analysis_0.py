from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/negative_leading_statement_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw intergroup contest dataframe to produce the variables used in modeling.

    Produces:
      - SizeDiff: n_focal - n_other (continuous)
      - SizeRatio: n_focal / n_other (continuous)
      - FocalLarger: binary indicator (1 if focal > other, 0 otherwise)
      - MaleDiff: m_focal - m_other (continuous)
      - FemaleDiff: f_focal - f_other (continuous)
      - DistAdv: dist_other - dist_focal (continuous); positive => focal is closer to its center relative to other

    Also drops rows with missing values in the required columns.
    """
    # make a copy
    df = df.copy()

    # Required columns for the analysis
    required_cols = [
        'win', 'n_focal', 'n_other',
        'dist_focal', 'dist_other',
        'm_focal', 'm_other', 'f_focal', 'f_other',
        'focal', 'other'
    ]

    # Drop rows with missing values in any required column
    df = df.dropna(subset=required_cols)

    # Compute relative size measures
    df['SizeDiff'] = df['n_focal'] - df['n_other']
    # avoid division by zero (dataset minimum n_other is >=5 by schema, but guard anyway)
    df['SizeRatio'] = df['n_focal'] / df['n_other'].replace({0: np.nan})
    df['FocalLarger'] = (df['SizeDiff'] > 0).astype(int)

    # Sex composition differences
    df['MaleDiff'] = df['m_focal'] - df['m_other']
    df['FemaleDiff'] = df['f_focal'] - df['f_other']

    # Location advantage: positive -> focal closer to its home-range center than the other group is to theirs
    df['DistAdv'] = df['dist_other'] - df['dist_focal']

    # Convert focal and other to categorical for modeling
    df['focal'] = df['focal'].astype('category')
    df['other'] = df['other'].astype('category')

    # Optional: standardize primary continuous predictors to aid model convergence/interpretation
    # (we keep the raw columns for interpretability; z-scored versions can be added if desired)
    df['SizeDiff_z'] = (df['SizeDiff'] - df['SizeDiff'].mean()) / df['SizeDiff'].std(ddof=0)
    df['DistAdv_z'] = (df['DistAdv'] - df['DistAdv'].mean()) / df['DistAdv'].std(ddof=0)

    # Ensure binary dependent variable is integer 0/1
    df['win'] = df['win'].astype(int)

    # Return the dataframe with all added columns (original columns retained)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a logistic regression (binomial GLM) predicting probability that the focal group wins.

    Formula:
      win ~ SizeDiff + DistAdv + MaleDiff + FemaleDiff + SizeDiff:DistAdv + C(focal) + C(other)

    Returns a dictionary with the fitted model object, odds ratios, and confidence intervals on the odds ratio scale,
    plus the full model summary object for inspection.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Work on a copy
    df = df.copy()

    # Ensure categorical variables are categories (should have been done in transform, but safe)
    df['focal'] = df['focal'].astype('category')
    df['other'] = df['other'].astype('category')

    # Primary formula: include interaction to test whether the effect of size depends on location
    formula = 'win ~ SizeDiff + DistAdv + MaleDiff + FemaleDiff + SizeDiff:DistAdv + C(focal) + C(other)'

    # Fit binomial GLM (logistic regression)
    glm_binom = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Compute odds ratios and confidence intervals
    params = glm_binom.params
    conf_int = glm_binom.conf_int()
    odds_ratios = np.exp(params)
    conf_int_odds = np.exp(conf_int)

    results = {
        'model': glm_binom,
        'odds_ratios': odds_ratios,
        'conf_int_odds': conf_int_odds,
        'summary': glm_binom.summary()
    }

    return results


