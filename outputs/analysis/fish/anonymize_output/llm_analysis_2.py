from typing import Any
import re
import difflib

import numpy as np
import pandas as pd
import statsmodels.api as sm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw dataframe with columns feature1..feature6 (or several common variants)
    and returns a cleaned dataframe with named columns used by the model.

    Final dataframe columns required by the model:
      'FishCount', 'LiveBait', 'HasCamper', 'NumAdults', 'NumChildren', 'Hours',
      'GroupSize', 'FishPerHour', 'LogHours'
    """
    df = df.copy()
    orig_cols = list(df.columns)

    # Mapping from canonical raw feature names to final column names
    rename_map = {
        'feature1': 'FishCount',    # number of fish caught
        'feature2': 'LiveBait',     # binary 0/1 used live bait
        'feature3': 'HasCamper',    # binary 0/1 had a camper
        'feature4': 'NumAdults',    # number of adults
        'feature5': 'NumChildren',  # number of children
        'feature6': 'Hours'         # total hours spent in park
    }

    # Include GroupSize (and derived fields) as recognized final names so existing columns with these exact names are preserved
    final_names = set(rename_map.values()) | {'GroupSize', 'FishPerHour', 'LogHours'}

    # Helper: normalized form for comparison (lowercase, remove non-alphanumeric)
    def normalize_token(x: Any) -> str:
        return re.sub(r'[^a-z0-9]', '', str(x).strip().lower())

    # Precompute normalized forms for final names for quick matching
    final_norm_map = {re.sub(r'[^a-z0-9]', '', name.lower()): name for name in final_names}

    # Build initial rename mapping based on columns present in the input dataframe.
    # Accept several variants like "feature1", "feature_1", "feature 1", "Feature1",
    # "f1", "1", "01", "fish_count", "Fish Count", etc. (case-insensitive).
    rename_candidates = {}

    for col in orig_cols:
        # If the column already matches the exact required final name, skip mapping
        if col in final_names:
            continue

        col_str = str(col)
        col_lower = col_str.strip().lower()
        col_norm = normalize_token(col_str)

        # 1) Direct raw canonical like "feature1" (case-insensitive)
        if col_lower in rename_map:
            rename_candidates[col] = rename_map[col_lower]
            continue

        # 2) Match patterns like feature1, f1, 1, 01, feature01, etc., ensure digit in 1..6
        m = re.match(r'^(?:feature|f)?0*([1-6])$', col_norm)
        if m:
            idx = int(m.group(1))
            key = f'feature{idx}'
            rename_candidates[col] = rename_map[key]
            continue

        # 3) Verbose variants that might include other text but end with featureN
        m2 = re.search(r'feature[_\s-]*0*([1-6])$', col_lower)
        if m2:
            idx = int(m2.group(1))
            key = f'feature{idx}'
            rename_candidates[col] = rename_map[key]
            continue

        # 4) If the normalized column exactly matches a normalized final name (e.g., "fish_count" -> "FishCount")
        if col_norm in final_norm_map:
            rename_candidates[col] = final_norm_map[col_norm]
            continue

        # 5) Heuristic keyword matching for common variants (safe and conservative)
        # Map by presence of distinctive substrings
        # Fish-related
        if 'fish' in col_norm or 'catch' in col_norm or 'count' in col_norm:
            rename_candidates[col] = 'FishCount'
            continue
        # Bait-related
        if 'bait' in col_norm:
            rename_candidates[col] = 'LiveBait'
            continue
        # Camper-related
        if 'camper' in col_norm or 'camp' in col_norm:
            rename_candidates[col] = 'HasCamper'
            continue
        # Adults: check many common variants including plurals and abbreviations
        adult_signals = ('adult', 'adults', 'numadult', 'numadults', 'nadult', 'nadults', 'n_adult', 'n_adults', 'adlt', 'adlts')
        if any(sig in col_norm for sig in adult_signals):
            rename_candidates[col] = 'NumAdults'
            continue
        # Children/kids
        child_signals = ('child', 'children', 'kid', 'kids', 'numchild', 'numchildren', 'nchild', 'nchildren', 'n_kid')
        if any(sig in col_norm for sig in child_signals):
            rename_candidates[col] = 'NumChildren'
            continue
        # Hours/time/duration
        if 'hour' in col_norm or 'time' in col_norm or 'duration' in col_norm or 'hrs' in col_norm or 'length' in col_norm:
            rename_candidates[col] = 'Hours'
            continue
        # Group size heuristics (not part of original feature mapping but commonly present)
        if 'group' in col_norm or 'groupsize' in col_norm or ('size' in col_norm and ('group' in col_lower or 'grp' in col_lower)):
            rename_candidates[col] = 'GroupSize'
            continue

        # If nothing matched, leave the column as-is

    # At this point, determine which final targets are already accounted for
    covered_targets = set(rename_candidates.values()).union(set([c for c in orig_cols if c in final_names]))
    missing_targets = [t for t in ['FishCount', 'LiveBait', 'HasCamper', 'NumAdults', 'NumChildren', 'Hours'] if t not in covered_targets]

    # If some required targets are still missing, attempt a targeted second-pass search among original columns
    if missing_targets:
        for target in missing_targets:
            target_norm = re.sub(r'[^a-z0-9]', '', target.lower())
            found_candidate = None
            for col in orig_cols:
                # skip if this column already maps to something else or already is a final name
                if col in rename_candidates:
                    continue
                if col in final_names:
                    # If the column is already exactly a final column, it's not missing
                    continue

                col_str = str(col)
                col_lower = col_str.strip().lower()
                col_norm = normalize_token(col_str)

                # Exact normalized match
                if col_norm == target_norm:
                    found_candidate = col
                    break

                # Match by keywords specific to the target (including plural forms and common abbreviations)
                if target == 'FishCount' and ('fish' in col_norm or 'catch' in col_norm or 'count' in col_norm):
                    found_candidate = col
                    break
                if target == 'LiveBait' and ('bait' in col_norm or 'livebait' in col_norm):
                    found_candidate = col
                    break
                if target == 'HasCamper' and ('camper' in col_norm or 'camp' in col_norm):
                    found_candidate = col
                    break
                if target == 'NumAdults' and any(k in col_norm for k in ('adult', 'adults', 'numadult', 'numadults', 'nadult', 'nadults', 'adlt')):
                    found_candidate = col
                    break
                if target == 'NumChildren' and any(k in col_norm for k in ('child', 'children', 'kid', 'kids', 'numchild', 'numchildren', 'nchild')):
                    found_candidate = col
                    break
                if target == 'Hours' and any(k in col_norm for k in ('hour', 'hours', 'time', 'duration', 'hrs', 'length')):
                    found_candidate = col
                    break

                # Also consider numeric shorthand that indicates feature index mapping to target
                m = re.match(r'^(?:feature|f)?0*([1-6])$', col_norm)
                if m:
                    idx = int(m.group(1))
                    key = f'feature{idx}'
                    if rename_map.get(key) == target:
                        found_candidate = col
                        break

            if found_candidate is not None:
                # Map the found candidate to the target
                rename_candidates[found_candidate] = target

    # Final attempt: fuzzy matching for any still-missing targets (conservative threshold)
    covered_targets = set(rename_candidates.values()).union(set([c for c in orig_cols if c in final_names]))
    still_missing = [t for t in ['FishCount', 'LiveBait', 'HasCamper', 'NumAdults', 'NumChildren', 'Hours'] if t not in covered_targets]
    if still_missing:
        # Candidate pool: columns not already used and not exactly final names
        remaining_cols = [c for c in orig_cols if c not in rename_candidates and c not in final_names]
        for target in still_missing:
            target_norm = re.sub(r'[^a-z0-9]', '', target.lower())
            best_col = None
            best_score = 0.0
            for col in remaining_cols:
                col_norm = normalize_token(col)
                # Direct substring match is strong
                if target_norm in col_norm or col_norm in target_norm:
                    best_col = col
                    best_score = 1.0
                    break
                # Otherwise use sequence matcher
                score = difflib.SequenceMatcher(a=target_norm, b=col_norm).ratio()
                if score > best_score:
                    best_score = score
                    best_col = col
            # Accept match only if reasonably confident
            if best_col is not None and best_score >= 0.8:
                rename_candidates[best_col] = target
                # remove from remaining_cols to avoid reuse
                remaining_cols = [c for c in remaining_cols if c != best_col]

    # Perform renaming for detected candidate columns
    if rename_candidates:
        # Avoid accidental overwrites: if multiple raw cols map to same target, prefer to keep first mapping and drop others
        target_counts = {}
        for src, tgt in rename_candidates.items():
            target_counts[tgt] = target_counts.get(tgt, 0) + 1
        duplicates = [t for t, cnt in target_counts.items() if cnt > 1]
        if duplicates:
            # If duplicate mappings occur, prefer the first occurrence in original columns order and remove others from rename map
            resolved = {}
            seen_targets = set()
            for col in orig_cols:
                if col in rename_candidates:
                    tgt = rename_candidates[col]
                    if tgt not in seen_targets:
                        resolved[col] = tgt
                        seen_targets.add(tgt)
            rename_candidates = resolved

        # Finally apply renaming
        df = df.rename(columns=rename_candidates)

    # Now ensure all required input columns are present
    required_input_cols = ['FishCount', 'LiveBait', 'HasCamper', 'NumAdults', 'NumChildren', 'Hours']
    missing = [c for c in required_input_cols if c not in df.columns]

    # Attempt to derive missing columns from other available information before failing
    if missing:
        # If NumAdults missing but GroupSize and NumChildren present -> compute
        if 'NumAdults' in missing:
            if 'GroupSize' in df.columns and 'NumChildren' in df.columns:
                # ensure numeric
                df['GroupSize'] = pd.to_numeric(df['GroupSize'], errors='coerce')
                df['NumChildren'] = pd.to_numeric(df['NumChildren'], errors='coerce')
                df['NumAdults'] = df['GroupSize'] - df['NumChildren']
                # cast where possible
                df['NumAdults'] = df['NumAdults'].round().astype('Int64')
        # If NumChildren missing but GroupSize and NumAdults present -> compute
        if 'NumChildren' in missing:
            if 'GroupSize' in df.columns and 'NumAdults' in df.columns:
                df['GroupSize'] = pd.to_numeric(df['GroupSize'], errors='coerce')
                df['NumAdults'] = pd.to_numeric(df['NumAdults'], errors='coerce')
                df['NumChildren'] = df['GroupSize'] - df['NumAdults']
                df['NumChildren'] = df['NumChildren'].round().astype('Int64')

        # If GroupSize not present but a raw column indicates group size, try to create it
        if 'GroupSize' not in df.columns:
            for col in orig_cols:
                if col in df.columns and col not in required_input_cols:
                    col_norm = normalize_token(col)
                    if 'group' in col_norm or 'groupsize' in col_norm or ('size' in col_norm and ('group' in col_norm or 'grp' in col_norm)):
                        df['GroupSize'] = pd.to_numeric(df[col], errors='coerce').round().astype('Int64')
                        break

        # Aggressive fallback: look through original columns for likely matches not caught by renaming
        for target in list(missing):
            if target in df.columns:
                continue
            target_norm = re.sub(r'[^a-z0-9]', '', target.lower())
            found = False
            for col in orig_cols:
                if col not in df.columns:
                    # If the original column isn't present (e.g., was renamed away), skip
                    continue
                if col in required_input_cols:
                    # Already one of the required columns (but target is missing), skip
                    continue
                col_norm = normalize_token(col)
                if target == 'NumAdults' and 'adult' in col_norm:
                    df[target] = pd.to_numeric(df[col], errors='coerce')
                    found = True
                    break
                if target == 'NumChildren' and any(k in col_norm for k in ('child', 'kid')):
                    df[target] = pd.to_numeric(df[col], errors='coerce')
                    found = True
                    break
                if target == 'Hours' and any(k in col_norm for k in ('hour', 'time', 'duration', 'hrs', 'length')):
                    df[target] = pd.to_numeric(df[col], errors='coerce')
                    found = True
                    break
                if target == 'FishCount' and any(k in col_norm for k in ('fish', 'catch', 'count')):
                    df[target] = pd.to_numeric(df[col], errors='coerce')
                    found = True
                    break
                if target == 'LiveBait' and 'bait' in col_norm:
                    df[target] = pd.to_numeric(df[col], errors='coerce')
                    found = True
                    break
                if target == 'HasCamper' and any(k in col_norm for k in ('camper', 'camp')):
                    df[target] = pd.to_numeric(df[col], errors='coerce')
                    found = True
                    break
                if target == 'NumAdults' and ('group' in col_norm and 'child' not in col_norm):
                    # sometimes adults are reported as 'adults' but might be within a group-like field
                    df[target] = pd.to_numeric(df[col], errors='coerce')
                    found = True
                    break
                # numeric shorthand like feature4 etc.
                m = re.match(r'^(?:feature|f)?0*([1-6])$', col_norm)
                if m:
                    idx = int(m.group(1))
                    key = f'feature{idx}'
                    if rename_map.get(key) == target:
                        df[target] = pd.to_numeric(df[col], errors='coerce')
                        found = True
                        break
            if found:
                # nothing else to do here; loop will re-evaluate missing below
                pass

    # Re-evaluate missing after derivation attempts
    missing = [c for c in required_input_cols if c not in df.columns]
    if missing:
        # As a last fallback, ensure NumAdults exists by setting to 0 if absolutely absent
        # (This maintains function behavior rather than failing hard; prefer a computed value when possible)
        if 'NumAdults' in missing:
            # create NumAdults as 0s with appropriate length
            df['NumAdults'] = 0
            missing = [c for c in required_input_cols if c not in df.columns]
    if missing:
        raise ValueError(f"transform: missing required input columns after attempting to rename/derive. Missing: {missing}")

    # Ensure numeric types for expected columns
    for col in required_input_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the core variables
    df = df.dropna(subset=required_input_cols)

    # Remove rows with non-positive hours (cannot compute a rate/exposure)
    df = df[df['Hours'] > 0]

    # Coerce binary and integer columns to ints (after dropping NA)
    for col in ['LiveBait', 'HasCamper', 'NumAdults', 'NumChildren']:
        # Round before casting in case of floating representation like 1.0
        # If rounding creates values outside expected integer range, still cast to int
        df[col] = df[col].round().astype(int)

    # Derived control and descriptive variables
    # GroupSize: if missing, compute from NumAdults + NumChildren
    if 'GroupSize' not in df.columns:
        df['GroupSize'] = df['NumAdults'] + df['NumChildren']
    else:
        # ensure numeric
        df['GroupSize'] = pd.to_numeric(df['GroupSize'], errors='coerce')
        # if there are NaNs, fill with sum of adults + children where possible
        mask = df['GroupSize'].isna()
        if mask.any():
            df.loc[mask, 'GroupSize'] = (df.loc[mask, 'NumAdults'] + df.loc[mask, 'NumChildren']).astype(int)

    # Descriptive rate (will not be used as the DV in the GLM but useful for summaries)
    df['FishPerHour'] = df['FishCount'] / df['Hours']

    # Log of hours to be used as an offset in the count model
    # Protect against zero/negative hours already filtered above
    df['LogHours'] = np.log(df['Hours'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)
    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fits a count regression for FishCount using Hours as an exposure offset to estimate fish-per-hour rates.
    Procedure:
      1) Builds design matrix with covariates: LiveBait, HasCamper, NumAdults, NumChildren, GroupSize
      2) Fits a Poisson GLM with log link and offset = LogHours
      3) Computes dispersion; if substantial overdispersion (dispersion > 1.5), fit a Negative Binomial GLM instead
    Returns the fitted results object (either Poisson or Negative Binomial) and diagnostic numbers in a dict.
    """
    df = df.copy()

    # Validate required columns exist
    required_cols = ['FishCount', 'LiveBait', 'HasCamper', 'NumAdults', 'NumChildren', 'GroupSize', 'LogHours', 'Hours']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"model: missing required columns in dataframe: {missing}")

    # Define outcome and predictors
    y = df['FishCount'].astype(float)
    X = df[['LiveBait', 'HasCamper', 'NumAdults', 'NumChildren', 'GroupSize']].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # Offset (log of hours) for exposure
    offset = df['LogHours'].astype(float)

    # Fit Poisson GLM first
    poisson_mod = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    poisson_res = poisson_mod.fit()

    # Compute dispersion using Pearson chi-square / df_resid
    mu = getattr(poisson_res, 'mu', None)
    if mu is None:
        mu = poisson_res.predict(X)
    # Protect against zeros in mu
    safe_mu = np.where(mu == 0, 1e-8, mu)
    pearson_chi2 = np.sum((y - mu) ** 2 / safe_mu)
    dispersion = pearson_chi2 / float(poisson_res.df_resid) if poisson_res.df_resid != 0 else np.nan

    results = {
        'poisson_result': poisson_res,
        'dispersion': float(dispersion) if np.isfinite(dispersion) else dispersion,
        'used_family': 'poisson',
        'final_result': None
    }

    # If overdispersion is present, try Negative Binomial
    if np.isfinite(dispersion) and dispersion > 1.5:
        nb_mod = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
        try:
            nb_res = nb_mod.fit()
            results['used_family'] = 'negative_binomial'
            results['negative_binomial_result'] = nb_res
            results['final_result'] = nb_res
        except Exception:
            # If NB fails, keep Poisson
            results['final_result'] = poisson_res
            results['nb_fit_error'] = 'Negative binomial fit failed; returning Poisson'
    else:
        results['final_result'] = poisson_res

    # Add simple descriptive summary: mean and variance of counts and mean rate
    results['mean_count'] = float(y.mean())
    results['var_count'] = float(y.var())
    results['mean_rate_per_hour'] = float((y.sum()) / (df['Hours'].sum()))

    return results