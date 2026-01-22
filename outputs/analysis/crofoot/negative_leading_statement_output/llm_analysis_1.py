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
    Transform the raw dataset into the set of variables used in the statistical model.

    Creates the following columns used in the model:
    - Win: binary outcome (int)
    - SizeAdv: n_focal - n_other (raw)
    - MaleAdv: m_focal - m_other (raw)
    - LocationAdv: dist_other - dist_focal (raw: positive => focal closer to its center)
    - SizeAdv_s, MaleAdv_s, LocationAdv_s: z-scored versions (mean 0, sd 1)
    - Dyad: categorical dyad id

    Drops rows with missing values in any columns required for these computations.
    """
    df = df.copy()

    # Ensure required columns exist and drop rows with missing values in them
    required_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'dyad']
    df = df.dropna(subset=required_cols)

    # Dependent variable
    df['Win'] = df['win'].astype(int)

    # Independent / control raw variables
    df['SizeAdv'] = df['n_focal'] - df['n_other']
    df['MaleAdv'] = df['m_focal'] - df['m_other']
    # Define LocationAdv so that positive = contest closer to focal group's center (focal advantage)
    df['LocationAdv'] = df['dist_other'] - df['dist_focal']

    # Standardize continuous predictors (z-score). Use population std (ddof=0) to avoid small-sample ddof issues.
    for col in ['SizeAdv', 'LocationAdv', 'MaleAdv']:
        sd = df[col].std(ddof=0)
        if sd == 0 or np.isnan(sd):
            # If there's no variance, create a zero column to avoid division by zero
            df[col + '_s'] = 0.0
        else:
            df[col + '_s'] = (df[col] - df[col].mean()) / sd

    # Dyad as categorical fixed-effect variable in the model
    df['Dyad'] = df['dyad'].astype('category')

    # Return only columns needed for modeling plus a few raw columns for diagnostics
    keep_cols = ['Win', 'SizeAdv', 'SizeAdv_s', 'LocationAdv', 'LocationAdv_s', 'MaleAdv', 'MaleAdv_s', 'Dyad',
                 'focal', 'other', 'dyad', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other']
    # Some of the keep_cols may not exist if the input lacked them; intersect with df.columns
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a binomial (logistic) regression predicting Win from SizeAdv, LocationAdv and their interaction,
    controlling for MaleAdv and dyad fixed effects.

    Formula: Win ~ SizeAdv_s * LocationAdv_s + MaleAdv_s + C(Dyad)

    Returns a dictionary with the fitted model object and odds ratios with confidence intervals for interpretation.
    """
    import statsmodels.formula.api as smf

    # Ensure the required columns are present
    required = ['Win', 'SizeAdv_s', 'LocationAdv_s', 'MaleAdv_s', 'Dyad']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Fit binomial GLM (logistic regression) with dyad as a categorical fixed effect
    formula = 'Win ~ SizeAdv_s * LocationAdv_s + MaleAdv_s + C(Dyad)'
    fit = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Compute odds ratios and 95% CI by exponentiating coefficients and CI bounds
    params = fit.params
    conf = fit.conf_int()
    or_df = pd.DataFrame({
        'coef': params,
        'OR': np.exp(params),
        'CI_lower': np.exp(conf.iloc[:, 0]),
        'CI_upper': np.exp(conf.iloc[:, 1])
    })

    results = {
        'model': fit,
        'odds_ratios': or_df,
        'aic': fit.aic,
        'bic': fit.bic,
        'deviance': fit.deviance,
        'n_obs': int(fit.nobs)
    }

    return results


