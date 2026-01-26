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
    # Work on a copy
    df = df.copy()

    # Drop rows missing critical variables needed for modeling
    df = df.dropna(subset=['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'f_focal', 'f_other'])

    # Relative size measures
    df['SizeDiff'] = df['n_focal'] - df['n_other']
    # Ratio as an alternative scale (avoid division by zero; n_other has min 5 in schema)
    df['SizeRatio'] = df['n_focal'] / df['n_other']

    # Location advantage: positive means focal is closer to its home-range center than the other
    df['DistDiff'] = df['dist_other'] - df['dist_focal']

    # Sex composition differences
    df['MaleDiff'] = df['m_focal'] - df['m_other']
    df['FemaleDiff'] = df['f_focal'] - df['f_other']

    # Total size control
    df['TotalSize'] = df['n_focal'] + df['n_other']

    # Standardize continuous predictors (z-score) for interpretation and numerical stability
    cont_cols = ['SizeDiff', 'SizeRatio', 'DistDiff', 'MaleDiff', 'FemaleDiff', 'TotalSize']
    for col in cont_cols:
        mu = df[col].mean()
        sigma = df[col].std(ddof=0)
        # If sigma is zero (unlikely here), produce zeros to avoid division by zero
        if sigma == 0 or np.isnan(sigma):
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mu) / sigma

    # Ensure dyad is categorical for use as a fixed effect in the model
    df['dyad'] = df['dyad'].astype('category')

    # Keep only columns needed for modeling plus originals useful for diagnostics
    keep_cols = [
        'focal', 'other', 'dyad', 'win',
        'n_focal', 'n_other', 'SizeDiff', 'SizeRatio', 'DistDiff',
        'm_focal', 'm_other', 'MaleDiff', 'f_focal', 'f_other', 'FemaleDiff',
        'TotalSize',
        'SizeDiff_z', 'SizeRatio_z', 'DistDiff_z', 'MaleDiff_z', 'FemaleDiff_z', 'TotalSize_z'
    ]

    # Some columns may not exist if original df had fewer columns; intersect to avoid KeyError
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """Fit a logistic regression (GLM with binomial family) predicting focal win.

    The model tests: main effects of relative size (SizeDiff_z) and location advantage (DistDiff_z),
    their interaction (SizeDiff_z:DistDiff_z), and controls for male/female composition differences,
    total contest size, and dyad fixed effects.

    Returns a dictionary containing the fitted model, odds ratios (with 95% CI), and the model summary text.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns present
    required = ['win', 'SizeDiff_z', 'DistDiff_z', 'MaleDiff_z', 'FemaleDiff_z', 'TotalSize_z', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Formulate model: include interaction to test whether location moderates size effect
    formula = 'win ~ SizeDiff_z * DistDiff_z + MaleDiff_z + FemaleDiff_z + TotalSize_z + C(dyad)'

    # Fit GLM with binomial family (logistic regression)
    model_fit = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit()

    # Compute odds ratios and 95% CI
    params = model_fit.params
    conf = model_fit.conf_int()
    or_series = np.exp(params)
    ci_lower = np.exp(conf[0])
    ci_upper = np.exp(conf[1])
    or_df = pd.DataFrame({'OR': or_series, '2.5%': ci_lower, '97.5%': ci_upper})

    # Prepare human-readable summary
    summary_text = model_fit.summary().as_text()

    # Return results for downstream inspection
    return {
        'model_fit': model_fit,
        'odds_ratios': or_df,
        'summary_text': summary_text
    }


