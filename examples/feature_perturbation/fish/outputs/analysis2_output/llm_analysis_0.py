from typing import Any
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe. Key steps:
      - map/rename input columns to required final column names
      - drop rows with missing or invalid values for key variables
      - coerce types
      - compute derived variables: TotalPeople, FishPerHour, log_hours (for offset)

    Required final columns (must be present in returned dataframe):
      ['FishCount','LiveBait','Camper','NumAdults','NumChildren','Hours','TotalPeople','FishPerHour','log_hours']
    """
    df = df.copy()

    # Helper to normalize column names for matching
    def _norm(col_name: str) -> str:
        return re.sub(r'[^a-z0-9_]', '', col_name.lower())

    # Candidate patterns for each target column (common variants)
    candidates = {
        'FishCount': ['fishcount', 'fish_count', 'fish', 'count', 'feature1', 'feature_1'],
        'LiveBait': ['livebait', 'live_bait', 'bait', 'feature2', 'feature_2'],
        'Camper': ['camper', 'has_camper', 'feature3', 'feature_3'],
        'NumAdults': ['numadults', 'num_adults', 'adults', 'adult', 'feature4', 'feature_4'],
        'NumChildren': ['numchildren', 'num_children', 'children', 'child', 'kids', 'feature5', 'feature_5'],
        'Hours': ['hours', 'hour', 'time', 'duration', 'visit_time', 'feature6', 'feature_6'],
    }

    # Find the best matching source column for each required target
    mapping = {}
    df_cols_norm = {col: _norm(col) for col in df.columns}

    for target, pats in candidates.items():
        found = None
        # 1) exact match on normalized name
        for col, col_norm in df_cols_norm.items():
            if col_norm in pats:
                found = col
                break
        # 2) contains any pattern
        if found is None:
            for col, col_norm in df_cols_norm.items():
                for pat in pats:
                    if pat in col_norm:
                        found = col
                        break
                if found is not None:
                    break
        if found is not None:
            mapping[found] = target

    # If mapping did not find all required targets, try a positional fallback
    required_targets = ['FishCount', 'LiveBait', 'Camper', 'NumAdults', 'NumChildren', 'Hours']
    missing_targets = [t for t in required_targets if t not in mapping.values()]
    if missing_targets:
        # If the dataframe has columns named like feature1..feature6 (anycase), map them by number
        feature_by_number = {}
        for col in df.columns:
            m = re.match(r'feature[_\s]*([0-9]+)$', col.strip().lower())
            if m:
                feature_by_number[int(m.group(1))] = col
        if feature_by_number:
            # map 1->FishCount, 2->LiveBait, 3->Camper, 4->NumAdults, 5->NumChildren, 6->Hours
            order_map = {1: 'FishCount', 2: 'LiveBait', 3: 'Camper', 4: 'NumAdults', 5: 'NumChildren', 6: 'Hours'}
            for num, col in feature_by_number.items():
                if num in order_map:
                    mapping[col] = order_map[num]

    # Final fallback: if still missing and there are at least 6 columns, assume first 6 correspond in the canonical order.
    missing_targets = [t for t in required_targets if t not in mapping.values()]
    if missing_targets and len(df.columns) >= 6:
        # determine which targets are still unmapped, map them in canonical order to the first N unmapped columns
        canonical_order = ['FishCount', 'LiveBait', 'Camper', 'NumAdults', 'NumChildren', 'Hours']
        # select columns that are not already used in mapping
        unused_cols = [c for c in df.columns if c not in mapping]
        for target, col in zip(canonical_order, unused_cols[:6]):
            if target not in mapping.values():
                mapping[col] = target

    # Apply renaming if any mapping found
    if mapping:
        df = df.rename(columns=mapping)

    # After initial renaming, attempt to infer NumAdults or NumChildren from any 'persons' / total people column if needed.
    # Recompute normalized column names
    df_cols_norm = {col: _norm(col) for col in df.columns}

    # Identify possible total-people columns (common variants)
    total_people_patterns = {'persons', 'personscount', 'totalpeople', 'total_people', 'total', 'group_size', 'groupsize', 'group'}
    total_col = None
    for col, col_norm in df_cols_norm.items():
        if col_norm in total_people_patterns:
            total_col = col
            break
        # also consider names that contain 'persons' or 'total' or 'group' as fallback
        if any(p in col_norm for p in ['persons', 'totalpeople', 'total', 'groupsize', 'group']):
            total_col = total_col or col

    # If NumAdults missing but NumChildren and total_col present, compute NumAdults = total - NumChildren
    if 'NumAdults' not in df.columns and 'NumChildren' in df.columns and total_col is not None:
        # coerce to numeric for computation
        df[total_col] = pd.to_numeric(df[total_col], errors='coerce')
        df['NumChildren'] = pd.to_numeric(df['NumChildren'], errors='coerce')
        # Compute adults where possible
        computed_adults = (df[total_col] - df['NumChildren']).fillna(0)
        # Negative values set to 0, round toward zero and convert to int
        computed_adults = computed_adults.clip(lower=0).astype(int)
        df['NumAdults'] = computed_adults

    # If NumChildren missing but NumAdults and total_col present, compute NumChildren = total - NumAdults
    if 'NumChildren' not in df.columns and 'NumAdults' in df.columns and total_col is not None:
        df[total_col] = pd.to_numeric(df[total_col], errors='coerce')
        df['NumAdults'] = pd.to_numeric(df['NumAdults'], errors='coerce')
        computed_children = (df[total_col] - df['NumAdults']).fillna(0)
        computed_children = computed_children.clip(lower=0).astype(int)
        df['NumChildren'] = computed_children

    # If both NumAdults and NumChildren missing but total_col present, assume all are adults (NumChildren=0)
    if 'NumAdults' not in df.columns and 'NumChildren' not in df.columns and total_col is not None:
        df[total_col] = pd.to_numeric(df[total_col], errors='coerce')
        df['NumAdults'] = df[total_col].fillna(0).astype(int)
        df['NumChildren'] = 0

    # Verify that all required final columns are present (before numeric coercion)
    final_required = ['FishCount', 'LiveBait', 'Camper', 'NumAdults', 'NumChildren', 'Hours']
    missing_after = [c for c in final_required if c not in df.columns]
    if missing_after:
        raise KeyError(f"Could not find required input columns in the dataframe. Missing: {missing_after}. "
                       f"Available columns: {list(df.columns)}")

    # Convert required columns to numeric where appropriate
    numeric_cols = final_required
    for c in numeric_cols:
        # Keep original column if integer already, else coerce
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing values in the required numeric inputs
    df = df.dropna(subset=numeric_cols)

    # Remove rows with non-positive hours (cannot take log for offset)
    df = df[df['Hours'] > 0]

    # Ensure binary 0/1 for LiveBait and Camper: treat any positive value as 1, else 0
    df['LiveBait'] = (df['LiveBait'] > 0).astype(int)
    df['Camper'] = (df['Camper'] > 0).astype(int)

    # Ensure integer counts for people (round down/truncate if necessary)
    df['NumAdults'] = df['NumAdults'].astype(int)
    df['NumChildren'] = df['NumChildren'].astype(int)

    # Derived columns
    df['TotalPeople'] = (df['NumAdults'] + df['NumChildren']).astype(int)
    df['FishPerHour'] = df['FishCount'] / df['Hours']
    df['log_hours'] = np.log(df['Hours'])

    # Final column order (keeps required columns and any extras)
    keep_cols = ['FishCount', 'LiveBait', 'Camper', 'NumAdults', 'NumChildren', 'Hours', 'TotalPeople', 'FishPerHour', 'log_hours']
    # If any of these are missing somehow (should not be), raise
    missing_final = [c for c in keep_cols if c not in df.columns]
    if missing_final:
        raise KeyError(f"After transformation, the following required final columns are missing: {missing_final}")

    return df[keep_cols]


def model(df: pd.DataFrame) -> Any:
    """
    Fit a count model for FishCount using Hours as exposure (offset) to estimate fish-per-hour rates.

    Steps:
      1. Fit Poisson GLM with offset = log_hours.
      2. Calculate dispersion (Pearson chi-square / df_resid). If substantial overdispersion is present (dispersion > 1.5),
         fit a Negative Binomial GLM and return it as the preferred model.

    Returns a dict containing the fitted models and diagnostics:
      {
        'poisson_model': <GLMResultsWrapper for Poisson>,
        'dispersion': <float>,
        'neg_bin_model': <GLMResultsWrapper for NegativeBinomial> or None
      }
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['FishCount', 'LiveBait', 'Camper', 'NumAdults', 'NumChildren', 'TotalPeople', 'log_hours']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Model input dataframe is missing required columns: {missing}")

    # Design matrix: main predictors and controls
    X = df[['LiveBait', 'Camper', 'NumAdults', 'NumChildren', 'TotalPeople']]
    X = sm.add_constant(X, has_constant='add')
    y = pd.to_numeric(df['FishCount'], errors='coerce').astype(int)
    offset = df['log_hours'].to_numpy()

    # Poisson model with offset (log Hours) to model rate per hour
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()

    # Compute dispersion: Pearson chi2 / df_resid. Values >>1 indicate overdispersion.
    pearson_chi2 = (poisson_model.resid_pearson ** 2).sum()
    dispersion = pearson_chi2 / poisson_model.df_resid if poisson_model.df_resid > 0 else np.nan

    result = {
        'poisson_model': poisson_model,
        'dispersion': dispersion,
        'neg_bin_model': None
    }

    # If overdispersed, fit Negative Binomial
    if not np.isnan(dispersion) and dispersion > 1.5:
        try:
            nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset).fit()
            result['neg_bin_model'] = nb_model
        except Exception:
            # If NegativeBinomial via GLM fails, leave neg_bin_model as None
            result['neg_bin_model'] = None

    return result