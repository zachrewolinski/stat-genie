from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/replace_with_rvs_output/crofoot.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modelling.

    Produces the following new columns used in the model:
      - RelSize_prop: focal proportion of total adult group size n_focal / (n_focal + n_other)
      - RelSize_std: z-scored RelSize_prop
      - Location: categorical ('FocalHome', 'OtherHome', 'Neutral') based on which group's home-center the contest is closer to
      - Location_Focal: binary indicator (1 if Location == 'FocalHome', else 0)
      - MaleAdv: m_focal - m_other
      - FemaleAdv: f_focal - f_other
      - DistDiff: dist_other - dist_focal (positive when focal is closer)

    Also drops rows with missing values in key columns used for derivations.
    """
    # work on a copy
    df = df.copy()

    # Drop rows missing any of the essential raw columns
    required_cols = [
        'win', 'dist_focal', 'dist_other',
        'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other', 'dyad'
    ]
    df = df.dropna(subset=required_cols)

    # Ensure numeric types
    numeric_cols = ['dist_focal', 'dist_other', 'n_focal', 'n_other', 'm_focal', 'm_other', 'f_focal', 'f_other']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=numeric_cols)

    # Relative size: proportion of total adults represented by focal group
    # (avoids scale issues if groups vary in absolute size)
    total_size = df['n_focal'] + df['n_other']
    # Guard against division by zero, though biologically sizes > 0 expected
    total_size = total_size.replace({0: np.nan})
    df['RelSize_prop'] = df['n_focal'] / total_size

    # Standardize relative size for model stability / interpretability
    df['RelSize_std'] = (df['RelSize_prop'] - df['RelSize_prop'].mean()) / df['RelSize_prop'].std(ddof=0)

    # Location: which group's home-range center the contest is closer to
    # If dist_focal < dist_other -> contest closer to focal home
    # If dist_other < dist_focal -> contest closer to other home
    # If (almost) equal (within 1 meter) -> Neutral
    def loc_label(row):
        if row['dist_focal'] + 1e-6 < row['dist_other'] - 1.0:
            return 'FocalHome'
        elif row['dist_other'] + 1e-6 < row['dist_focal'] - 1.0:
            return 'OtherHome'
        else:
            return 'Neutral'

    df['Location'] = df.apply(loc_label, axis=1)
    df['Location_Focal'] = (df['Location'] == 'FocalHome').astype(int)

    # Composition advantage controls
    df['MaleAdv'] = df['m_focal'] - df['m_other']
    df['FemaleAdv'] = df['f_focal'] - df['f_other']

    # Continuous distance difference control: positive if focal is closer to its center
    df['DistDiff'] = df['dist_other'] - df['dist_focal']

    # Standardize continuous controls (except dyad / Location)
    for col in ['MaleAdv', 'FemaleAdv', 'DistDiff']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col + '_std'] = (df[col] - df[col].mean()) / (df[col].std(ddof=0) if df[col].std(ddof=0) != 0 else 1.0)
        # Keep the raw difference columns for clarity in results too

    # Keep only columns needed for the modelling stage (plus a few raw columns for traceability)
    keep_cols = [
        'focal', 'other', 'dyad', 'win',
        'dist_focal', 'dist_other',
        'n_focal', 'n_other', 'RelSize_prop', 'RelSize_std',
        'Location', 'Location_Focal',
        'm_focal', 'm_other', 'MaleAdv', 'MaleAdv_std',
        'f_focal', 'f_other', 'FemaleAdv', 'FemaleAdv_std',
        'DistDiff', 'DistDiff_std'
    ]
    # It's okay if some of these are missing; intersect with existing columns
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a logistic regression to estimate how relative group size and contest location
    affect the probability that the focal group wins an intergroup contest.

    Model specification:
      win ~ RelSize_std * Location_Focal + MaleAdv_std + FemaleAdv_std + DistDiff_std + C(dyad)

    - Interaction tests whether the effect of relative size differs when the contest is near the focal group's home-range center.
    - Dyad entered as a categorical control (fixed effects). We also compute cluster-robust standard errors clustered by dyad.

    Returns the fitted model results object with cluster-robust covariances.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns are present
    required_model_cols = ['win', 'RelSize_std', 'Location_Focal', 'MaleAdv_std', 'FemaleAdv_std', 'DistDiff_std', 'dyad']
    missing = [c for c in required_model_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modelling: {missing}")

    # Build formula: include interaction between relative size and focal-location advantage
    formula = 'win ~ RelSize_std * Location_Focal + MaleAdv_std + FemaleAdv_std + DistDiff_std + C(dyad)'

    # Fit logistic regression (binomial logit)
    fit_res = smf.logit(formula=formula, data=df).fit(disp=False)

    # Obtain cluster-robust standard errors clustered by dyad
    try:
        results = fit_res.get_robustcov_results(cov_type='cluster', groups=df['dyad'])
    except Exception:
        # If clustering fails for any reason, return the plain fit
        results = fit_res

    return results


