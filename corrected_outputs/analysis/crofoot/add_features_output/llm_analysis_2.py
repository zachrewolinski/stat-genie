from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/crofoot/add_features_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw capuchin intergroup contest data into analysis-ready columns.

    Output columns used in the model:
      - win: binary outcome (unchanged)
      - size_log_ratio: log(n_focal / n_other)
      - CloserToFocal: binary indicator (1 if dist_other - dist_focal > 0, else 0)
      - male_diff: m_focal - m_other
      - dist_diff: dist_other - dist_focal (continuous)
      - dyad: categorical dyad id

    The function drops rows with missing values in any of the required columns.
    """
    df = df.copy()

    # Required columns for the analysis
    required = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other', 'dyad']
    # Drop rows with missing values in required columns
    df = df.dropna(subset=required)

    # Ensure numeric types for the numeric columns
    numeric_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other', 'm_focal', 'm_other']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=numeric_cols)  # drop if coercion produced NaN

    # Dependent variable: ensure binary integer
    # Coerce any non-0/1 to integers; assume input is already 0/1 or truthy/falsy
    df['win'] = df['win'].astype(int)

    # Independent variable: relative group size as log ratio
    eps = 1e-6
    df['size_log_ratio'] = np.log((df['n_focal'] + eps) / (df['n_other'] + eps))

    # Location measures
    # dist_diff > 0 indicates contest is relatively closer to the focal group's center
    df['dist_diff'] = df['dist_other'] - df['dist_focal']
    df['CloserToFocal'] = (df['dist_diff'] > 0).astype(int)

    # Control: male composition difference
    df['male_diff'] = df['m_focal'] - df['m_other']

    # Cast dyad to category for use as a fixed-effect / clustering variable
    df['dyad'] = df['dyad'].astype('category')

    # Keep the columns necessary for modeling plus original counts for transparency
    keep_cols = ['win', 'size_log_ratio', 'CloserToFocal', 'male_diff', 'dist_diff', 'dyad',
                 'n_focal', 'n_other', 'm_focal', 'm_other', 'dist_focal', 'dist_other']
    df = df[keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression predicting the probability the focal group wins.

    Model formula:
      win ~ size_log_ratio * CloserToFocal + male_diff + dist_diff + C(dyad)

    This fits a main effect of relative group size and contest location (binary), their interaction,
    and controls for male composition difference and continuous distance contrast. Dyad is included
    as a categorical fixed effect to account for pair-specific baseline differences. We obtain
    cluster-robust standard errors clustered by dyad to account for non-independence within dyads.

    Returns:
      results: fitted results object with cluster-robust covariance (use .summary() to inspect)
    """
    # Ensure required columns are present
    required_cols = ['win', 'size_log_ratio', 'CloserToFocal', 'male_diff', 'dist_diff', 'dyad']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Use logistic regression (equivalent to binomial GLM with logit link)
    formula = 'win ~ size_log_ratio * CloserToFocal + male_diff + dist_diff + C(dyad)'

    # Use the dyad labels as the cluster grouping
    clusters = df['dyad'].values

    # Build the model
    logit_mod = smf.logit(formula=formula, data=df)

    # Fit the model and request cluster-robust covariance from the fitter
    # This avoids relying on methods that may not exist on the results object.
    results = logit_mod.fit(disp=False, cov_type='cluster', cov_kwds={'groups': clusters})

    return results