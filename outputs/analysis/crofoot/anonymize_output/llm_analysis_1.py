from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the raw dataset to the modeling dataframe. The function:
    - renames columns to meaningful names
    - coerces to numeric, fills missing raw columns with NaN
    - creates derived predictors: RelSize, RelSize_log, LocAdv, LocCategory, MaleDiff, FemaleDiff, TotalSize
    - z-standardizes continuous predictors used in the model
    - preserves group and dyad identifiers for clustered SEs / fixed effects

    Final dataframe columns used in the model:
      - Win, RelSize_z, LocAdv_z, MaleDiff_z, FemaleDiff_z, TotalSize_z, DyadID, FocalGroupID, OtherGroupID
    """
    df = df.copy()

    # rename raw feature columns to descriptive names (if those raw columns exist)
    rename_map = {
        'feature1': 'FocalGroupID',
        'feature2': 'OtherGroupID',
        'feature3': 'DyadID',
        'feature4': 'Win',            # 1 if focal won, 0 if other won
        'feature5': 'FocalDist',      # distance focal from center of its home range (m)
        'feature6': 'OtherDist',      # distance other from center of its home range (m)
        'feature7': 'FocalSize',
        'feature8': 'OtherSize',
        'feature9': 'FocalMales',
        'feature10': 'OtherMales',
        'feature11': 'FocalFemales',
        'feature12': 'OtherFemales'
    }

    # Only rename columns that are present to avoid KeyError
    existing_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing_rename:
        df = df.rename(columns=existing_rename)

    # Ensure all expected raw columns (after rename) exist in the dataframe; if missing, create them as NaN
    for col in rename_map.values():
        if col not in df.columns:
            df[col] = np.nan

    # ensure numeric types where appropriate (coerce non-numeric to NaN)
    numeric_cols = ['Win', 'FocalDist', 'OtherDist', 'FocalSize', 'OtherSize',
                    'FocalMales', 'OtherMales', 'FocalFemales', 'OtherFemales']
    # Do not coerce ID columns to numeric here to preserve string IDs if present
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # If Win is not strictly 0/1, try to coerce common truthy values
    if 'Win' in df.columns:
        # If Win is boolean, convert to int; if numeric already but not 0/1, leave as-is (GLM will error later if inappropriate)
        if df['Win'].dtype == bool:
            df['Win'] = df['Win'].astype(int)

    # drop rows with missing essential variables (only those columns are considered essential)
    essential = ['Win', 'FocalDist', 'OtherDist', 'FocalSize', 'OtherSize']
    df = df.dropna(subset=essential)

    # derived predictors
    df['RelSize'] = df['FocalSize'] - df['OtherSize']
    # log-ratio (small offset to avoid log(0)) for robustness / sensitivity checks
    df['RelSize_log'] = np.log((df['FocalSize'] + 0.5) / (df['OtherSize'] + 0.5))

    # Location advantage: other_dist - focal_dist. Positive => contest closer to focal group's center => advantage focal
    df['LocAdv'] = df['OtherDist'] - df['FocalDist']

    # a coarse categorical indicator of location (not used in primary model but useful descriptively)
    # threshold chosen as 10 meters to identify near-neutral locations; can be adjusted
    df['LocCategory'] = np.where(df['LocAdv'] > 10, 'FocalHome',
                                 np.where(df['LocAdv'] < -10, 'OtherHome', 'Neutral'))
    df['FocalHomeAdv'] = (df['LocAdv'] > 0).astype(int)

    # demographic differences
    df['MaleDiff'] = df['FocalMales'] - df['OtherMales']
    df['FemaleDiff'] = df['FocalFemales'] - df['OtherFemales']

    # absolute control: total size of both groups
    df['TotalSize'] = df['FocalSize'] + df['OtherSize']

    # standardize continuous predictors used in the model (z-scores) to aid interpretation and numerical stability
    stdize_cols = ['RelSize', 'LocAdv', 'MaleDiff', 'FemaleDiff', 'TotalSize']
    for col in stdize_cols:
        # compute mean and std on non-missing values
        series = df[col]
        mean = series.mean()
        std = series.std(ddof=0)
        # guard against zero std or all-NaN
        if std == 0 or np.isnan(std):
            # If there is no variation or no data, create a zero column to avoid downstream errors.
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (series - mean) / std

    # keep only columns required for modeling (but keep extras for checks)
    # final columns we will rely on: Win, RelSize_z, LocAdv_z, MaleDiff_z, FemaleDiff_z, TotalSize_z, DyadID, FocalGroupID, OtherGroupID
    required_cols = [
        'Win', 'RelSize_z', 'LocAdv_z', 'MaleDiff_z', 'FemaleDiff_z', 'TotalSize_z',
        'DyadID', 'FocalGroupID', 'OtherGroupID',
        # extras for checks / diagnostics
        'LocCategory', 'RelSize', 'RelSize_log', 'LocAdv', 'MaleDiff', 'FemaleDiff', 'TotalSize'
    ]
    # ensure all required columns exist (some may be missing in degenerate datasets)
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Reorder columns sensibly: put core modeling cols first
    col_order = ['Win', 'RelSize_z', 'LocAdv_z', 'MaleDiff_z', 'FemaleDiff_z', 'TotalSize_z',
                 'DyadID', 'FocalGroupID', 'OtherGroupID'] + [c for c in df.columns if c not in
                                                               ['Win', 'RelSize_z', 'LocAdv_z', 'MaleDiff_z',
                                                                'FemaleDiff_z', 'TotalSize_z', 'DyadID',
                                                                'FocalGroupID', 'OtherGroupID']]
    # remove duplicates while preserving order
    seen = set()
    ordered_cols = []
    for c in col_order:
        if c not in seen:
            ordered_cols.append(c)
            seen.add(c)

    df = df.loc[:, ordered_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fits a logistic regression (GLM with binomial family) predicting probability that the focal group wins.
    Primary predictors: standardized relative group size and standardized location advantage, including their interaction.
    Controls: standardized male difference, female difference, and total group size.
    Returns both the GLM result object and a clustered-robust covariance result clustered by DyadID.

    Model formula (primary): Win ~ RelSize_z * LocAdv_z + MaleDiff_z + FemaleDiff_z + TotalSize_z
    """
    import statsmodels.api as _sm
    import statsmodels.formula.api as _smf

    df = df.copy()

    # drop any rows missing the model variables
    model_vars = ['Win', 'RelSize_z', 'LocAdv_z', 'MaleDiff_z', 'FemaleDiff_z', 'TotalSize_z', 'DyadID']
    df_model = df.dropna(subset=model_vars)

    # ensure DyadID is present (it may be numeric or string); clustering can work with either
    if 'DyadID' in df_model.columns:
        # keep as-is; but if it's all NaN or empty DataFrame, we'll handle below
        pass

    # if there are no observations left after dropping missing values, return gracefully
    formula = 'Win ~ RelSize_z * LocAdv_z + MaleDiff_z + FemaleDiff_z + TotalSize_z'
    if df_model.shape[0] == 0:
        return {
            'glm_result': None,
            'clustered_result': None,
            'formula': formula,
            'n_obs': 0
        }

    # coerce Win to numeric (0/1) if possible
    try:
        df_model['Win'] = pd.to_numeric(df_model['Win'], errors='coerce')
    except Exception:
        pass
    # final check: drop rows where Win could not be coerced
    df_model = df_model.dropna(subset=['Win'])
    if df_model.shape[0] == 0:
        return {
            'glm_result': None,
            'clustered_result': None,
            'formula': formula,
            'n_obs': 0
        }

    # fit GLM (logistic)
    glm_res = None
    try:
        glm_res = _smf.glm(formula=formula, data=df_model, family=_sm.families.Binomial()).fit()
    except Exception:
        # If fitting fails for any reason, return a safe object indicating failure
        return {
            'glm_result': None,
            'clustered_result': None,
            'formula': formula,
            'n_obs': df_model.shape[0]
        }

    # compute clustered robust standard errors by DyadID (accounts for non-independence within dyads)
    clustered_res = None
    try:
        clustered_res = glm_res.get_robustcov_results(cov_type='cluster', groups=df_model['DyadID'])
    except Exception:
        # fallback: supply heteroskedasticity-robust (HC3) if clustering fails
        try:
            clustered_res = glm_res.get_robustcov_results(cov_type='HC3')
        except Exception:
            clustered_res = None

    # return the fitted objects for downstream inspection
    return {
        'glm_result': glm_res,
        'clustered_result': clustered_res,
        'formula': formula,
        'n_obs': df_model.shape[0]
    }