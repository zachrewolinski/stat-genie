from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/noperturb_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw capuchin contest dataset into a dataframe containing the variables used in modeling.

    Produces the following new columns (returned in the dataframe):
      - SizeAdv = n_focal - n_other
      - DistAdv = dist_other - dist_focal
      - FocalCloser = 1 if dist_focal < dist_other else 0
      - MaleAdv = m_focal - m_other
      - FemaleAdv = f_focal - f_other
      - TotalN = n_focal + n_other
      - size- and location-centered versions: SizeAdv_c, DistAdv_c, MaleAdv_c, FemaleAdv_c, TotalN_c
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Required columns
    required_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
                     'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad']

    # Drop rows missing any required columns
    df = df.dropna(subset=required_cols)

    # Ensure numeric types where relevant
    numeric_cols = ['win', 'n_focal', 'n_other', 'dist_focal', 'dist_other',
                    'm_focal', 'm_other', 'f_focal', 'f_other']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop again if conversion produced NaNs
    df = df.dropna(subset=numeric_cols)

    # Derived variables
    df['SizeAdv'] = df['n_focal'] - df['n_other']
    # Positive DistAdv means focal is closer to its center than other (focal location advantage)
    df['DistAdv'] = df['dist_other'] - df['dist_focal']
    df['FocalCloser'] = (df['dist_focal'] < df['dist_other']).astype(int)
    df['MaleAdv'] = df['m_focal'] - df['m_other']
    df['FemaleAdv'] = df['f_focal'] - df['f_other']
    df['TotalN'] = df['n_focal'] + df['n_other']

    # Mean-center continuous predictors for interpretability
    df['SizeAdv_c'] = df['SizeAdv'] - df['SizeAdv'].mean()
    df['DistAdv_c'] = df['DistAdv'] - df['DistAdv'].mean()
    df['MaleAdv_c'] = df['MaleAdv'] - df['MaleAdv'].mean()
    df['FemaleAdv_c'] = df['FemaleAdv'] - df['FemaleAdv'].mean()
    df['TotalN_c'] = df['TotalN'] - df['TotalN'].mean()

    # Keep only columns necessary for modeling (but retain dyad and identifying columns)
    model_cols = ['focal', 'other', 'dyad', 'win', 'SizeAdv_c', 'DistAdv_c', 'FocalCloser',
                  'MaleAdv_c', 'FemaleAdv_c', 'TotalN_c',
                  # keep raw versions for possible diagnostics
                  'SizeAdv', 'DistAdv', 'MaleAdv', 'FemaleAdv', 'TotalN']

    existing_model_cols = [c for c in model_cols if c in df.columns]
    df = df[existing_model_cols]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression to predict the probability that the focal group wins (win = 1).

    Primary model tests:
      - main effect of size advantage (SizeAdv_c)
      - main effect of location advantage (DistAdv_c)
      - interaction of size advantage with being closer to home (SizeAdv_c * FocalCloser)
    Controls: MaleAdv_c, FemaleAdv_c, TotalN_c

    Uses cluster-robust standard errors clustered on dyad to account for non-independence within dyads.

    Returns:
      - results: statsmodels results object with clustered robust covariance (when possible)
    """
    # Ensure required variables exist
    required = ['win', 'SizeAdv_c', 'DistAdv_c', 'FocalCloser', 'MaleAdv_c', 'FemaleAdv_c', 'TotalN_c', 'dyad']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Fit logistic regression (logit)
    # Interaction between size advantage and FocalCloser tests whether location (being closer) moderates the size effect
    formula = 'win ~ SizeAdv_c * FocalCloser + DistAdv_c + MaleAdv_c + FemaleAdv_c + TotalN_c'

    # Try to get cluster-robust covariance by specifying cov_type at fit time.
    # If clustering fails (e.g., due to invalid groups), fall back to heteroskedasticity-robust (HC1).
    try:
        results = smf.logit(formula, data=df).fit(disp=False, cov_type='cluster', cov_kwds={'groups': df['dyad']})
    except Exception:
        results = smf.logit(formula, data=df).fit(disp=False, cov_type='HC1')

    return results