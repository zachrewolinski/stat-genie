from typing import Any
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform original dataset to variables required for modeling.

    Input expected columns (from schema):
      - feature1: number of fish caught
      - feature2: whether used livebait (0/1)
      - feature3: whether group had a camper (0/1)
      - feature4: number of adults
      - feature5: number of children
      - feature6: number of hours spent in park

    Returns dataframe with columns used by the model:
      - fish_count, used_livebait, has_camper, adults, children, hours, group_size,
        fish_per_hour, log_hours

    The function is robust to alternate raw column names by attempting to
    locate common variants (case/spacing/underscore variants) and renaming
    them to the required final column names.
    """
    df = df.copy()

    # Mapping of required final names to possible raw column names that might appear
    candidate_map = {
        'fish_count': ['feature1', 'fish_count', 'fish', 'num_fish', 'n_fish', 'fishcaught', 'fish_caught'],
        'used_livebait': ['feature2', 'used_livebait', 'livebait', 'used_live_bait', 'usedbait'],
        'has_camper': ['feature3', 'has_camper', 'camper', 'has_camper_present', 'campers'],
        'adults': ['feature4', 'adults', 'num_adults', 'n_adults'],
        'children': ['feature5', 'children', 'num_children', 'kids', 'childrens'],
        'hours': ['feature6', 'hours', 'time_hours', 'duration_hours', 'park_hours', 'hours_spent', 'time']
    }

    # Helper to normalize names for flexible matching (lowercase, remove non-alphanumeric)
    def _normalize(name: str) -> str:
        if not isinstance(name, str):
            return ''
        return re.sub(r'[^a-z0-9]', '', name.lower())

    # Build map of normalized existing column name -> actual column name
    norm_to_col = {}
    for col in df.columns:
        norm = _normalize(col)
        if norm and norm not in norm_to_col:
            norm_to_col[norm] = col

    # Build rename mapping by matching normalized candidate names to normalized existing columns.
    rename_map = {}
    used_originals = set()

    for final_name, candidates in candidate_map.items():
        # If final_name already exists exactly, skip renaming for it
        if final_name in df.columns:
            continue
        found = False

        # 1) Exact normalized candidate match
        for cand in candidates:
            norm_cand = _normalize(cand)
            if norm_cand in norm_to_col:
                original_col = norm_to_col[norm_cand]
                if original_col in used_originals:
                    continue
                rename_map[original_col] = final_name
                used_originals.add(original_col)
                found = True
                break
        if found:
            continue

        # 2) Match by normalized final name (someone may have provided close variant)
        norm_final = _normalize(final_name)
        if norm_final in norm_to_col:
            original_col = norm_to_col[norm_final]
            if original_col not in used_originals:
                rename_map[original_col] = final_name
                used_originals.add(original_col)
                continue

        # 3) Substring / token match: check if any normalized existing column contains any candidate token
        cand_tokens = [_normalize(c) for c in candidates if _normalize(c)]
        for col in df.columns:
            if col in used_originals or col in rename_map:
                continue
            norm_col = _normalize(col)
            if not norm_col:
                continue
            for token in cand_tokens:
                if token and (token in norm_col or norm_col in token):
                    rename_map[col] = final_name
                    used_originals.add(col)
                    found = True
                    break
            if found:
                break
        if found:
            continue

        # 4) Pattern match for featureN variants (Feature 1, feature_1, etc.)
        feature_match = re.match(r'^\s*feature[_\s-]*([0-9]+)\s*$', final_name, flags=re.IGNORECASE)
        # Note: final_name won't match this normally; instead we check candidates for 'featureN'
        if not found:
            for cand in candidates:
                if re.match(r'^\s*feature[_\s-]*([0-9]+)\s*$', cand, flags=re.IGNORECASE):
                    # Extract number
                    num_match = re.search(r'([0-9]+)', cand)
                    if num_match:
                        n = num_match.group(1)
                        pattern = rf'^\s*feature[_\s-]*{re.escape(n)}\s*$'
                        for col in df.columns:
                            if col in used_originals or col in rename_map:
                                continue
                            if re.match(pattern, str(col), flags=re.IGNORECASE):
                                rename_map[col] = final_name
                                used_originals.add(col)
                                found = True
                                break
                        if found:
                            break

    # Additional fallback: try to map common "featureN" patterns if not yet mapped
    fallback_patterns = {
        'fish_count': r'^\s*feature[_\s-]*1\s*$',
        'used_livebait': r'^\s*feature[_\s-]*2\s*$',
        'has_camper': r'^\s*feature[_\s-]*3\s*$',
        'adults': r'^\s*feature[_\s-]*4\s*$',
        'children': r'^\s*feature[_\s-]*5\s*$',
        'hours': r'^\s*feature[_\s-]*6\s*$'
    }
    for final_name, pattern in fallback_patterns.items():
        if final_name in df.columns:
            continue
        for col in df.columns:
            if col in rename_map:
                continue
            if col in used_originals:
                continue
            if re.match(pattern, str(col), flags=re.IGNORECASE):
                rename_map[col] = final_name
                used_originals.add(col)
                break

    # If there are still missing critical columns, attempt relaxed matching by keywords
    critical_keywords = {
        'fish_count': ['fish', 'caught', 'count'],
        'hours': ['hour', 'time', 'duration']
    }
    for final_name, keywords in critical_keywords.items():
        if final_name in df.columns:
            continue
        # if already mapped via rename_map skip
        mapped = False
        for orig, mapped_to in rename_map.items():
            if mapped_to == final_name:
                mapped = True
                break
        if mapped:
            continue
        for col in df.columns:
            if col in used_originals or col in rename_map:
                continue
            norm_col = _normalize(col)
            if not norm_col:
                continue
            for kw in keywords:
                if _normalize(kw) in norm_col:
                    rename_map[col] = final_name
                    used_originals.add(col)
                    mapped = True
                    break
            if mapped:
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    # After attempting to rename, ensure the two critical columns exist before using them.
    critical = ['fish_count', 'hours']
    missing_critical = [c for c in critical if c not in df.columns]
    if missing_critical:
        raise ValueError(f"Input dataframe is missing required columns required for transformation: {missing_critical}")

    # Keep only rows with non-missing count and positive hours (exposure must be > 0)
    df = df.dropna(subset=['fish_count', 'hours'])
    # Remove non-positive or extremely tiny hours (can't take log of <=0)
    df = df[pd.to_numeric(df['hours'], errors='coerce') > 0]

    # Ensure integer / numeric types where appropriate
    # For count and binary predictors, coerce to numeric; for missing set sensible defaults (0)
    df['fish_count'] = pd.to_numeric(df['fish_count'], errors='coerce').fillna(0).astype(int)

    # For predictors that may be missing entirely, create them as zeros (safe default) if absent after rename attempt
    for col in ['used_livebait', 'has_camper', 'adults', 'children']:
        if col not in df.columns:
            # create column of zeros to satisfy downstream expectations
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    df['hours'] = pd.to_numeric(df['hours'], errors='coerce')
    # Drop any rows that became NaN after coercion of hours
    df = df.dropna(subset=['hours'])

    # Derived variables
    df['group_size'] = df['adults'] + df['children']
    # Fish per hour (continuous rate) for summary / diagnostics
    df['fish_per_hour'] = df['fish_count'] / df['hours']
    # Log of hours used as offset in GLM (exposure)
    # Keep raw 'hours' as a control/exposure column; create log_hours for convenience
    # Guard against zero/negative hours has already been done; still protect numerically
    df['log_hours'] = np.log(df['hours'].replace({0: np.nan}))

    # Final columns (explicit ordering helps downstream code)
    final_cols = ['fish_count', 'hours', 'log_hours', 'fish_per_hour', 'used_livebait', 'has_camper', 'adults', 'children', 'group_size']
    # Select intersection in case some optional diagnostics are absent
    final_cols = [c for c in final_cols if c in df.columns]
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count model for fish caught using hours as exposure.

    Primary approach:
      1) Fit a Poisson GLM with a log-offset = log(hours).
      2) Compute dispersion (Pearson chi-square / df_resid). If dispersion > 1.5 (overdispersion),
         refit using a Negative Binomial GLM.

    Returns the fitted statsmodels result object (either Poisson or NegativeBinomial selected).
    Also prints model summaries and basic diagnostics.
    """
    # Required columns check (these are the final dataframe columns that must be present)
    required = ['fish_count', 'hours', 'log_hours', 'used_livebait', 'has_camper', 'group_size', 'adults', 'children']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Ensure numeric types for modeling
    df = df.copy()
    for col in required:
        if col in ['used_livebait', 'has_camper', 'group_size', 'adults', 'children']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        elif col in ['hours', 'log_hours']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif col == 'fish_count':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Drop rows with NaNs in critical model inputs
    df = df.dropna(subset=['fish_count', 'hours', 'log_hours', 'used_livebait', 'has_camper', 'group_size', 'adults', 'children'])

    # Exogenous matrix (predictors)
    X = df[['used_livebait', 'has_camper', 'group_size', 'adults', 'children']].copy()
    X = sm.add_constant(X, has_constant='add')
    y = df['fish_count']
    offset = df['log_hours']

    # Fit Poisson
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    poisson_results = poisson_model.fit()

    # Pearson chi2 dispersion estimate
    mu = poisson_results.mu  # fitted mean
    # protect against division by zero
    denom = np.where(mu > 0, mu, 1.0)
    pearson_chi2 = np.sum(((y - mu) ** 2) / denom)
    df_resid = poisson_results.df_resid if hasattr(poisson_results, 'df_resid') else (len(y) - X.shape[1])
    dispersion = pearson_chi2 / df_resid if df_resid > 0 else np.nan

    print('Poisson model fitted. Dispersion (Pearson chi2 / df_resid) =', dispersion)
    print(poisson_results.summary())

    # If overdispersed, fit Negative Binomial
    if (not np.isnan(dispersion)) and (dispersion > 1.5):
        print('Overdispersion detected (dispersion > 1.5). Fitting Negative Binomial model...')
        try:
            nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
            nb_results = nb_model.fit()
            print(nb_results.summary())
            return nb_results
        except Exception as e:
            print('Negative Binomial GLM failed; returning Poisson results. Error:', e)
            return poisson_results
    else:
        # Dispersion OK, return Poisson
        return poisson_results