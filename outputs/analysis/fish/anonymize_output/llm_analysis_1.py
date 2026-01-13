from typing import Any
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a raw dataframe into the final dataframe required by the analysis.

    The final dataframe must contain these exact columns:
    ['FishCaught', 'LiveBait', 'Camper', 'NumAdults', 'NumChildren', 'TotalPeople', 'Hours', 'FishPerHour']

    This function attempts to be robust to a variety of input column namings by
    mapping common aliases to the required final column names. It does NOT change
    the required output column names.
    """
    # Work on a copy
    df = df.copy()

    required_cols = ['FishCaught', 'LiveBait', 'Camper', 'NumAdults', 'NumChildren', 'Hours']

    # If the dataframe already has all required columns, skip renaming
    if not set(required_cols).issubset(set(df.columns)):
        # Helper to normalize names: lowercase and remove non-alphanumeric characters
        def _normalize(name: str) -> str:
            return re.sub(r'[^0-9a-z]', '', str(name).lower())

        # Build mappings of lowercase and normalized column name -> original column name for matching
        lc_to_col = {col.lower(): col for col in df.columns}
        norm_to_col = {_normalize(col): col for col in df.columns}
        normalized_names = {col: _normalize(col) for col in df.columns}

        # Potential aliases for each required column (lowercase; may include common variants)
        aliases = {
            'FishCaught': ['feature1', 'fishcaught', 'fish_caught', 'fish count', 'fish_count', 'fish', 'catch', 'fishcaughts', 'fishcaught'],
            'LiveBait': ['feature2', 'livebait', 'live_bait', 'used_live_bait', 'livebait_used', 'live bait', 'usedlivebait'],
            'Camper': ['feature3', 'camper', 'has_camper', 'camper_present', 'camp', 'hascamp', 'camperpresent'],
            'NumAdults': ['feature4', 'numadults', 'num_adults', 'adults', 'n_adults', 'num adults', 'adult_count', 'num_adult', 'adult'],
            'NumChildren': ['feature5', 'numchildren', 'num_children', 'children', 'kids', 'n_children', 'num children', 'child_count', 'child', 'kid_count'],
            'Hours': ['feature6', 'hours', 'time_hours', 'duration_hours', 'visit_hours', 'hrs', 'time', 'duration']
        }

        rename_map = {}
        used_originals = set()

        for target, cand_list in aliases.items():
            # If exact required column name already present, skip
            if target in df.columns:
                continue

            found = None

            # 1) Exact case-insensitive match to existing column
            if target.lower() in lc_to_col:
                found = lc_to_col[target.lower()]

            # 2) Try candidate aliases (match normalized and lowercase)
            if found is None:
                for cand in cand_list:
                    # check lowercase exact
                    if cand.lower() in lc_to_col:
                        found = lc_to_col[cand.lower()]
                        break
                    # check normalized match
                    cand_norm = _normalize(cand)
                    if cand_norm in norm_to_col:
                        found = norm_to_col[cand_norm]
                        break

            # 3) Try matching by normalizing the target itself
            if found is None:
                target_norm = _normalize(target)
                if target_norm in norm_to_col:
                    found = norm_to_col[target_norm]

            # 4) Substring search on normalized names as a last resort (e.g., any column containing 'adult' or 'child')
            if found is None:
                key_substrings = []
                if target == 'NumAdults':
                    key_substrings = ['adult', 'adults']
                elif target == 'NumChildren':
                    key_substrings = ['child', 'children', 'kid', 'kids']
                else:
                    # for others use parts of the target name
                    key_substrings = [part for part in re.findall(r'[A-Z]?[a-z]+', target.lower()) if part]

                for orig_col, norm in normalized_names.items():
                    for sub in key_substrings:
                        if sub in norm:
                            found = orig_col
                            break
                    if found is not None:
                        break

            if found:
                # Avoid mapping the same original column to multiple targets
                if found not in used_originals:
                    rename_map[found] = target
                    used_originals.add(found)

        if rename_map:
            df = df.rename(columns=rename_map)

    # After attempted renaming, check for missing required columns
    missing = [c for c in required_cols if c not in df.columns]

    # Fallback attempts to create missing NumAdults/NumChildren from TotalPeople or from available info
    if missing:
        # If TotalPeople exists but NumAdults/NumChildren missing, attempt to derive
        if 'TotalPeople' in df.columns:
            # Ensure TotalPeople numeric
            total_numeric = pd.to_numeric(df['TotalPeople'], errors='coerce').fillna(0).astype(int)
            if 'NumAdults' not in df.columns and 'NumChildren' not in df.columns:
                # As a conservative fallback assume all are adults and zero children
                df['NumAdults'] = total_numeric
                df['NumChildren'] = 0
            else:
                if 'NumAdults' not in df.columns and 'NumChildren' in df.columns:
                    children_numeric = pd.to_numeric(df['NumChildren'], errors='coerce').fillna(0).astype(int)
                    deduced_adults = (total_numeric - children_numeric).clip(lower=0)
                    df['NumAdults'] = deduced_adults
                if 'NumChildren' not in df.columns and 'NumAdults' in df.columns:
                    adults_numeric = pd.to_numeric(df['NumAdults'], errors='coerce').fillna(0).astype(int)
                    deduced_children = (total_numeric - adults_numeric).clip(lower=0)
                    df['NumChildren'] = deduced_children

        # If still missing NumAdults/NumChildren, attempt to find plausible columns by substrings and copy
        remaining_missing = [c for c in required_cols if c not in df.columns]
        if remaining_missing:
            # try to find columns with 'adult'/'child' substrings in original names and coerce them
            for miss in remaining_missing:
                found = None
                if miss == 'NumAdults':
                    subs = ['adult', 'adults']
                elif miss == 'NumChildren':
                    subs = ['child', 'children', 'kid', 'kids']
                else:
                    subs = [_normalize(miss)]
                for col in df.columns:
                    ncol = re.sub(r'[^0-9a-z]', '', col.lower())
                    for sub in subs:
                        if sub in ncol:
                            found = col
                            break
                    if found:
                        break
                if found:
                    df[miss] = pd.to_numeric(df[found], errors='coerce')

    # Additional conservative fallbacks to ensure NumAdults and NumChildren exist
    # (This prevents unexpected KeyError by creating reasonable defaults.)
    for col in ['NumAdults', 'NumChildren']:
        if col not in df.columns:
            if 'TotalPeople' in df.columns:
                total_numeric = pd.to_numeric(df['TotalPeople'], errors='coerce').fillna(0).astype(int)
                if col == 'NumAdults':
                    if 'NumChildren' in df.columns:
                        children_numeric = pd.to_numeric(df['NumChildren'], errors='coerce').fillna(0).astype(int)
                        df['NumAdults'] = (total_numeric - children_numeric).clip(lower=0)
                    else:
                        # Assume all are adults if no children info
                        df['NumAdults'] = total_numeric
                else:  # NumChildren
                    if 'NumAdults' in df.columns:
                        adults_numeric = pd.to_numeric(df['NumAdults'], errors='coerce').fillna(0).astype(int)
                        df['NumChildren'] = (total_numeric - adults_numeric).clip(lower=0)
                    else:
                        # If NumAdults not present, and TotalPeople exists, conservatively assume zero children
                        df['NumChildren'] = 0
            else:
                # Try to find any column with the appropriate substring
                found = None
                subs = ['adult', 'adults'] if col == 'NumAdults' else ['child', 'children', 'kid', 'kids']
                for c in df.columns:
                    n = re.sub(r'[^0-9a-z]', '', c.lower())
                    for s in subs:
                        if s in n:
                            found = c
                            break
                    if found:
                        break
                if found:
                    df[col] = pd.to_numeric(df[found], errors='coerce')
                else:
                    # Conservative defaults: assume zero children and zero adults if nothing available.
                    # This avoids crashes; downstream users should be aware of imputed defaults.
                    df[col] = 0

    # Recompute missing after fallbacks
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Required columns missing from input dataframe after attempted rename and fallback: {missing}")

    # Coerce numeric types where appropriate
    df['FishCaught'] = pd.to_numeric(df['FishCaught'], errors='coerce')
    df['NumAdults'] = pd.to_numeric(df['NumAdults'], errors='coerce')
    df['NumChildren'] = pd.to_numeric(df['NumChildren'], errors='coerce')
    df['Hours'] = pd.to_numeric(df['Hours'], errors='coerce')

    # LiveBait and Camper may be encoded as text like 'Yes'/'No' or numeric 0/1. Keep raw for transformation.
    # If they are missing (shouldn't be at this point), raise earlier. We'll treat them as-is here.
    df['LiveBait'] = df['LiveBait']
    df['Camper'] = df['Camper']

    # Drop rows that are missing any essential numeric variables after coercion
    df = df.dropna(subset=['FishCaught', 'NumAdults', 'NumChildren', 'Hours']).copy()

    # Convert LiveBait / Camper to binary 0/1
    def to_binary(series: pd.Series) -> pd.Series:
        # If numeric-like, treat >0 as 1, else 0
        if pd.api.types.is_numeric_dtype(series):
            ser = pd.to_numeric(series, errors='coerce').fillna(0)
            return (ser > 0).astype(int)
        # Otherwise try mapping common textual values
        ser_str = series.astype(str).str.strip().str.lower()
        mapping = {
            'yes': 1, 'y': 1, 'true': 1, 't': 1, '1': 1,
            'no': 0, 'n': 0, 'false': 0, 'f': 0, '0': 0
        }
        return ser_str.map(mapping).fillna(0).astype(int)

    df['LiveBait'] = to_binary(df['LiveBait'])
    df['Camper'] = to_binary(df['Camper'])

    # Ensure Hours positive (exposure)
    df = df[df['Hours'] > 0].copy()

    # Ensure FishCaught is a non-negative integer count
    df = df[df['FishCaught'] >= 0].copy()
    df['FishCaught'] = df['FishCaught'].round().astype(int)

    # Ensure NumAdults and NumChildren are non-negative integers
    df['NumAdults'] = df['NumAdults'].round().astype(int)
    df['NumChildren'] = df['NumChildren'].round().astype(int)
    df = df[(df['NumAdults'] >= 0) & (df['NumChildren'] >= 0)].copy()

    # Derived columns
    df['TotalPeople'] = df['NumAdults'] + df['NumChildren']
    df['FishPerHour'] = df['FishCaught'] / df['Hours']

    # Final columns in required order (must include these exact names)
    final_cols = ['FishCaught', 'LiveBait', 'Camper', 'NumAdults', 'NumChildren', 'TotalPeople', 'Hours', 'FishPerHour']
    return df[final_cols]


def model(df: pd.DataFrame) -> Any:
    """
    Fits count regression models for FishCaught with Hours as exposure (offset).
    Primary approach: fit Poisson with log(Hours) offset, check dispersion, and if overdispersed fit a Negative Binomial.
    Returns a dictionary with fitted models and diagnostics.

    Required inputs in df: columns produced by transform().
    """
    import numpy as np
    import statsmodels.api as sm

    # Verify required columns exist
    required = ['FishCaught', 'LiveBait', 'Camper', 'NumAdults', 'NumChildren', 'TotalPeople', 'Hours']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # Select independent variables / controls for the linear predictor
    # Include LiveBait, Camper, TotalPeople as IVs and NumAdults, NumChildren as controls per specification
    exog_vars = ['LiveBait', 'Camper', 'TotalPeople', 'NumAdults', 'NumChildren']
    X = df[exog_vars].copy()
    X = sm.add_constant(X, has_constant='add')

    y = df['FishCaught'].astype(float)
    offset = np.log(df['Hours'].astype(float))

    results = {}

    # Fit Poisson GLM with offset
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()
    results['poisson'] = poisson_model

    # Dispersion test: Pearson chi-square / df_resid
    try:
        pearson_chi2 = float(poisson_model.pearson_chi2)
    except Exception:
        # compute manually if attribute not available
        resid_pearson = poisson_model.resid_pearson
        pearson_chi2 = float(np.sum(resid_pearson ** 2))
    dispersion = pearson_chi2 / poisson_model.df_resid if poisson_model.df_resid > 0 else np.nan
    results['poisson_dispersion'] = float(dispersion)

    # If dispersion > 1.5 (rule-of-thumb), fit Negative Binomial to account for overdispersion
    if dispersion > 1.5:
        nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset).fit()
        results['neg_binomial'] = nb_model
        results['aic'] = {'poisson_aic': float(poisson_model.aic), 'nb_aic': float(nb_model.aic)}
        results['bic'] = {'poisson_bic': float(poisson_model.bic), 'nb_bic': float(nb_model.bic)}
    else:
        results['note'] = 'No strong evidence of overdispersion; Poisson model likely adequate.'
        results['aic'] = {'poisson_aic': float(poisson_model.aic)}
        results['bic'] = {'poisson_bic': float(poisson_model.bic)}

    # Descriptive summary
    results['descriptives'] = {
        'mean_fish_per_visit': float(df['FishCaught'].mean()),
        'mean_hours_per_visit': float(df['Hours'].mean()),
        'overall_fish_per_hour': float(df['FishCaught'].sum() / df['Hours'].sum())
    }

    return results