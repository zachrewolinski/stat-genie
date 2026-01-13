from typing import Any
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the final dataframe used for modeling.

    Input columns expected (from raw data) may vary in naming; this function
    attempts to recognize common variants and rename them to the canonical
    final column names required by the model.

    Returns a dataframe with columns (used by the model):
      - Won (0/1)
      - RelativeSize (focal_size - other_size)
      - RelativeSizeRatio (focal_size / other_size)
      - LocationRelative (other_distance - focal_distance; positive = focal closer)
      - LocationAdvBinary (1 if focal is relatively closer, else 0)
      - NumMalesDiff (focal_males - other_males)
      - NumFemalesDiff (focal_females - other_females)
      - FocalGroupID, OtherGroupID, DyadID
      - FocalDistance, OtherDistance, FocalSize, OtherSize, FocalMales, OtherMales, FocalFemales, OtherFemales
    """
    df = df.copy()

    # Mapping from canonical raw feature names to required final column names.
    # These canonical raw names are those used by the original pipeline.
    raw_to_final = {
        'feature1': 'FocalGroupID',
        'feature2': 'OtherGroupID',
        'feature3': 'DyadID',
        'feature4': 'Won',
        'feature5': 'FocalDistance',
        'feature6': 'OtherDistance',
        'feature7': 'FocalSize',
        'feature8': 'OtherSize',
        'feature9': 'FocalMales',
        'feature10': 'OtherMales',
        'feature11': 'FocalFemales',
        'feature12': 'OtherFemales'
    }

    required_final_cols = list(raw_to_final.values())

    # Helper: normalize column names (lowercase, remove non-alphanumeric)
    def _norm(s: str) -> str:
        return re.sub(r'\W+', '', str(s).lower())

    norm_cols = {col: _norm(col) for col in df.columns}

    # Build a mapping from existing column names to the required final names.
    rename_map = {}

    # Token map with common variants (kept concise to avoid overly strict matching)
    token_map = {
        'FocalGroupID': ['focal', 'group', 'id'],
        'OtherGroupID': ['other', 'group', 'id'],
        'DyadID': ['dyad', 'pair', 'pairid', 'pair_id', 'dyadid', 'dyad_id', 'dyad'],
        'Won': ['won', 'win', 'outcome', 'result', 'winner'],
        'FocalDistance': ['focal', 'dist', 'distance', 'home'],
        'OtherDistance': ['other', 'dist', 'distance', 'home'],
        'FocalSize': ['focal', 'n', 'size', 'count'],
        'OtherSize': ['other', 'n', 'size', 'count'],
        'FocalMales': ['focal', 'male', 'males', 'num_males', 'm'],
        'OtherMales': ['other', 'male', 'males', 'num_males', 'm'],
        'FocalFemales': ['focal', 'female', 'females', 'num_females', 'f'],
        'OtherFemales': ['other', 'female', 'females', 'num_females', 'f'],
    }

    # For each required final column, try to find the most plausible existing column.
    for final_col in required_final_cols:
        # If final column already present, nothing to do.
        if final_col in df.columns:
            continue

        mapped = False

        # 1) Check if the canonical raw name exists (feature1..feature12)
        raw_keys = [rk for rk, fv in raw_to_final.items() if fv == final_col]
        for rk in raw_keys:
            if rk in df.columns:
                rename_map[rk] = final_col
                mapped = True
                break
            # also try normalized match for raw key
            n_rk = _norm(rk)
            for col, ncol in norm_cols.items():
                if ncol == n_rk:
                    rename_map[col] = final_col
                    mapped = True
                    break
            if mapped:
                break
        if mapped:
            continue

        # 2) Try to find a column whose normalized form equals the normalized final name
        n_final = _norm(final_col)
        for col, ncol in norm_cols.items():
            if ncol == n_final:
                rename_map[col] = final_col
                mapped = True
                break
        if mapped:
            continue

        # 3) Use token-based heuristics for common variants
        tokens = token_map.get(final_col, [n_final])
        # normalize tokens
        tnorm = [re.sub(r'\W+', '', t.lower()) for t in tokens]

        # Define required number of token matches:
        # require at least 2 token matches when possible, else 1
        min_matches = 2 if len(set(tnorm)) >= 2 else 1

        for col, ncol in norm_cols.items():
            matches = sum(1 for t in tnorm if t and t in ncol)
            if matches >= min_matches:
                rename_map[col] = final_col
                mapped = True
                break
        if mapped:
            continue

        # 4) Last resort: look for columns that contain at least one meaningful token
        for col, ncol in norm_cols.items():
            if any(t in ncol for t in tnorm if t):
                rename_map[col] = final_col
                mapped = True
                break

    # Only keep rename mappings where source exists and is different from destination
    rename_map = {src: dst for src, dst in rename_map.items() if src in df.columns and src != dst}

    if rename_map:
        df = df.rename(columns=rename_map)
        # recompute normalized columns after renaming
        norm_cols = {col: _norm(col) for col in df.columns}

    # After attempted renaming, some required columns might still be missing.
    # If group IDs are missing but DyadID exists, we can infer group IDs deterministically
    # from DyadID so that the model has consistent categorical identifiers.
    missing_final = [c for c in required_final_cols if c not in df.columns]

    # If DyadID present but one or both group ID columns missing, create them.
    if 'DyadID' in df.columns:
        if 'FocalGroupID' not in df.columns:
            df['FocalGroupID'] = df['DyadID'].astype(str) + '_F'
        if 'OtherGroupID' not in df.columns:
            df['OtherGroupID'] = df['DyadID'].astype(str) + '_O'

    # Re-evaluate missing after potential creation
    missing_final = [c for c in required_final_cols if c not in df.columns]
    if missing_final:
        raise KeyError(
            "Input dataframe is missing required columns after attempting to rename. "
            f"Missing final columns: {missing_final}. "
            f"Available columns: {list(df.columns)}. "
            "Provide a dataframe with the required fields or with recognizable variant names."
        )

    # Drop rows with missing values in key columns required for modeling
    df = df.dropna(subset=required_final_cols)

    # Ensure numeric types where appropriate
    numeric_cols = [
        'Won', 'FocalDistance', 'OtherDistance', 'FocalSize', 'OtherSize',
        'FocalMales', 'OtherMales', 'FocalFemales', 'OtherFemales'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows that became NA after coercion for numeric columns that exist
    existing_numeric = [c for c in numeric_cols if c in df.columns]
    if existing_numeric:
        df = df.dropna(subset=existing_numeric)

    # Outcome as integer 0/1
    # If Won column exists, coerce to binary integers (assumes truthy wins like 1/'1'/True/'win')
    if 'Won' in df.columns:
        # Try common mappings: 'win'/'won'/'1'/1/True -> 1, otherwise 0
        def _coerce_won(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, (int, float, np.integer, np.floating)):
                return int(x != 0)
            s = str(x).strip().lower()
            if s in {'1', 'true', 't', 'yes', 'y', 'won', 'win'}:
                return 1
            if s in {'0', 'false', 'f', 'no', 'n', 'lost', 'lose'}:
                return 0
            # fallback: try numeric conversion
            try:
                return int(float(s) != 0)
            except Exception:
                return np.nan

        df['Won'] = df['Won'].apply(_coerce_won)
        df = df.dropna(subset=['Won'])
        df['Won'] = df['Won'].astype(int)

    # Relative size measures
    df['RelativeSize'] = df['FocalSize'] - df['OtherSize']
    # Ratio is useful as an alternative metric (guard against division by zero)
    df['RelativeSizeRatio'] = df['FocalSize'] / df['OtherSize'].replace({0: np.nan})

    # Location advantage: compute continuous contrast and a binary indicator
    # LocationRelative > 0 means the focal group is closer to its home center than the other group is to its own center
    df['LocationRelative'] = df['OtherDistance'] - df['FocalDistance']
    df['LocationAdvBinary'] = (df['LocationRelative'] > 0).astype(int)

    # Sex-composition differences
    df['NumMalesDiff'] = df['FocalMales'] - df['OtherMales']
    df['NumFemalesDiff'] = df['FocalFemales'] - df['OtherFemales']

    # Keep only columns needed for modeling plus describers
    keep_cols = [
        'Won', 'RelativeSize', 'RelativeSizeRatio', 'LocationRelative', 'LocationAdvBinary',
        'NumMalesDiff', 'NumFemalesDiff',
        'FocalGroupID', 'OtherGroupID', 'DyadID',
        'FocalDistance', 'OtherDistance', 'FocalSize', 'OtherSize',
        'FocalMales', 'OtherMales', 'FocalFemales', 'OtherFemales'
    ]
    df = df[keep_cols]

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a logistic regression (GLM with binomial family) predicting the probability the focal group wins.

    Primary predictors:
      - RelativeSize (continuous)
      - LocationAdvBinary (0/1)
      - Interaction RelativeSize * LocationAdvBinary to test whether location moderates the effect of relative size

    Controls:
      - NumMalesDiff, NumFemalesDiff
      - Categorical fixed effects for FocalGroupID and OtherGroupID

    Standard errors are clustered by DyadID to account for multiple observations of the same pair.

    Returns the fitted (robust) results object.
    """
    # Validate that required columns are present in the input dataframe
    required_cols = [
        'Won', 'RelativeSize', 'LocationAdvBinary', 'NumMalesDiff', 'NumFemalesDiff',
        'FocalGroupID', 'OtherGroupID', 'DyadID'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Model input dataframe is missing required columns: {missing}")

    # Formula: interaction between relative size and location advantage; adjust for sex-composition and group fixed effects
    formula = (
        'Won ~ RelativeSize * LocationAdvBinary + NumMalesDiff + NumFemalesDiff '
        '+ C(FocalGroupID) + C(OtherGroupID)'
    )

    # Fit GLM with binomial family
    glm_mod = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    glm_res = glm_mod.fit()

    # Obtain cluster-robust covariance (clustered by DyadID)
    try:
        res_robust = glm_res.get_robustcov_results(cov_type='cluster', groups=df['DyadID'])
    except Exception:
        # If clustering fails for any reason, fall back to the original fit
        res_robust = glm_res

    # Return the results object (you can call res_robust.summary() outside this function)
    return res_robust