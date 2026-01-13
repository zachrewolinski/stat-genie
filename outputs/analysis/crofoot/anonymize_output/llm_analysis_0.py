from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Optional top-level read (kept for compatibility with original file structure)
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/crofoot/anonymize_output/crofoot.csv')
except Exception:
    # If the file isn't available in the current environment, skip reading.
    df = pd.DataFrame()


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Record which columns were present in the original input to avoid
    # dropping all rows when we later create placeholder NaN columns.
    original_raw_columns = list(df.columns)

    # Capture any original raw series corresponding to the outcome before renaming/coercion
    raw_focalwin_series = None

    # Rename raw feature columns to descriptive column names used in the model when possible
    rename_map = {
        'feature1': 'FocalID',
        'feature2': 'OtherID',
        'feature3': 'DyadID',
        'feature4': 'FocalWin',
        'feature5': 'DistToFocal',
        'feature6': 'DistToOther',
        'feature7': 'FocalSize',
        'feature8': 'OtherSize',
        'feature9': 'FocalMales',
        'feature10': 'OtherMales',
        'feature11': 'FocalFemales',
        'feature12': 'OtherFemales'
    }

    # If the original raw dataframe had a column that maps to FocalWin, capture it now (prior to renaming)
    for raw_col in original_raw_columns:
        if raw_col in rename_map and rename_map[raw_col] == 'FocalWin':
            raw_focalwin_series = df[raw_col].copy()
            break

    # Also handle the case where the input already used the final name 'FocalWin'
    if raw_focalwin_series is None and 'FocalWin' in original_raw_columns:
        raw_focalwin_series = df['FocalWin'].copy()

    # If still not found, try to heuristically detect a plausible outcome column by name
    if raw_focalwin_series is None:
        # common keywords that often label outcome columns
        outcome_keywords = {'win', 'winner', 'outcome', 'result', 'victory'}
        for raw_col in original_raw_columns:
            low = raw_col.lower()
            # exact matches
            if low in {'focalwin', 'focal_win', 'focalwin_flag', 'focal_win_flag'}:
                raw_focalwin_series = df[raw_col].copy()
                break
            # contains a keyword like 'win' or 'outcome'
            if any(k in low for k in outcome_keywords):
                raw_focalwin_series = df[raw_col].copy()
                break

    # Only rename columns that actually exist in the input to avoid creating unexpected keys
    present_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    if present_rename:
        df = df.rename(columns=present_rename)

    # Build a set of final column names that correspond to columns that were present in the original raw input.
    # This maps raw names that were renamed to their new names, and also includes any original columns
    # that already matched final names.
    original_final_columns = set()
    for raw_col in original_raw_columns:
        if raw_col in rename_map:
            original_final_columns.add(rename_map[raw_col])
        else:
            original_final_columns.add(raw_col)

    # Define numeric columns we expect and coerce them if present; otherwise create them as NaN columns
    numeric_cols = ['FocalWin', 'DistToFocal', 'DistToOther', 'FocalSize', 'OtherSize',
                    'FocalMales', 'OtherMales', 'FocalFemales', 'OtherFemales']

    for c in numeric_cols:
        if c in df.columns:
            # If the column currently exists (either because it was present originally or because it was renamed),
            # coerce to numeric (may introduce NaN for bad values)
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            # Create missing numeric columns as NaN so subsequent operations don't KeyError
            df[c] = pd.Series(np.nan, index=df.index, dtype=float)

    # NOTE: Do not drop rows here; leave row filtering to the modeling step.
    # This avoids accidentally removing all rows when some essential columns are absent or mostly NA.

    # Outcome: ensure binary 0/1 if column exists; otherwise create empty float column
    if 'FocalWin' in df.columns:
        # First try to coerce to numeric; if that yields all-NA but original had strings, attempt to map common string values.
        df['FocalWin'] = pd.to_numeric(df['FocalWin'], errors='coerce')

        if df['FocalWin'].isna().all():
            # Try to map common textual encodings to 0/1
            def map_outcome(val):
                if pd.isna(val):
                    return np.nan
                s = str(val).strip().lower()
                if s in {'1', '1.0', 'true', 't', 'yes', 'y', 'win', 'won', 'focal', 'focal_win', 'focalwin'}:
                    return 1.0
                if s in {'0', '0.0', 'false', 'f', 'no', 'n', 'loss', 'lost', 'other', 'other_win', 'otherwin'}:
                    return 0.0
                # Try to parse numeric strings that to_numeric may have missed
                try:
                    num = float(s)
                    return float(num)
                except Exception:
                    return np.nan

            # If we captured a raw series prior to renaming/coercion, map that series.
            if raw_focalwin_series is not None:
                mapped = raw_focalwin_series.apply(map_outcome)
                df['FocalWin'] = pd.to_numeric(mapped, errors='coerce')
            else:
                # As a fallback, attempt to map using the string representation of current values (best-effort)
                mapped = df['FocalWin'].apply(lambda v: map_outcome(v) if pd.isna(v) else v)
                df['FocalWin'] = pd.to_numeric(mapped, errors='coerce')

        # Ensure numeric dtype (float) for statsmodels.
        df['FocalWin'] = df['FocalWin'].astype(float)
    else:
        df['FocalWin'] = pd.Series(dtype=float, index=df.index)

    # Relative size measures
    df['RelSize'] = df['FocalSize'] - df['OtherSize']

    # log ratio (alternative measure, kept for diagnostics if needed)
    # Use +1 to avoid log(0); if sizes are NaN result will be NaN
    df['LogSizeRatio'] = np.log((df['FocalSize'].fillna(0) + 1) / (df['OtherSize'].fillna(0) + 1))

    # Location measures: positive DistanceDiff means other is farther (contest closer to focal center)
    df['DistanceDiff'] = df['DistToOther'] - df['DistToFocal']

    # Categorical location: use a threshold to define whether contest is clearly in one group's range
    thresh = 50.0
    # If DistanceDiff is NaN, classify as 'Neutral' (safe default)
    df['Location'] = np.where(
        df['DistanceDiff'].notna() & (df['DistanceDiff'] >= thresh),
        'FocalRange',
        np.where(
            df['DistanceDiff'].notna() & (df['DistanceDiff'] <= -thresh),
            'OtherRange',
            'Neutral'
        )
    )

    # Male advantage as a control
    df['MaleAdvantage'] = df['FocalMales'] - df['OtherMales']

    # Standardize continuous predictors used in the model for interpretability
    for col in ['RelSize', 'DistanceDiff']:
        # Compute mean and std using available (non-NA) values
        mean = df[col].mean(skipna=True)
        std = df[col].std(ddof=0, skipna=True)
        if pd.isna(std) or std == 0:
            # If std is zero or undefined, create a zero-filled column with the same index
            df[col + '_z'] = pd.Series(0.0, index=df.index, dtype=float)
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Ensure DyadID exists (create if missing so downstream code can check and fail with informative message)
    if 'DyadID' not in df.columns:
        # create a DyadID that is unique per row to avoid clustering failures later.
        df['DyadID'] = pd.Series([f'dyad_missing_{i}' for i in range(len(df))], index=df.index, dtype=object)
    else:
        # If some DyadID values are missing, fill those with a unique per-row identifier to avoid losing rows during modeling.
        if df['DyadID'].isna().any():
            missing_mask = df['DyadID'].isna()
            df.loc[missing_mask, 'DyadID'] = [f'dyad_missing_{i}' for i in df[missing_mask].index]

    # Ensure final required columns exist even if they were not in original data,
    # so the output dataframe has a consistent schema (they may be all-NA).
    final_required = ['RelSize_z', 'DistanceDiff_z', 'FocalWin', 'Location', 'MaleAdvantage', 'DyadID']
    for col in final_required:
        if col not in df.columns:
            # Create empty/NA column with matching index and reasonable dtype
            if col in ['RelSize_z', 'DistanceDiff_z', 'FocalWin', 'MaleAdvantage']:
                df[col] = pd.Series(np.nan, index=df.index, dtype=float)
            else:
                df[col] = pd.Series(pd.NA, index=df.index)

    # Fill any missing MaleAdvantage values with 0.0 as a neutral default
    # This avoids dropping rows in modeling when male counts are not reported for some contests.
    df['MaleAdvantage'] = df['MaleAdvantage'].fillna(0.0).astype(float)

    # As a final safety, ensure Location is non-null for all rows
    df['Location'] = df['Location'].fillna('Neutral')

    # If FocalWin is entirely missing but there's an indicator that other columns might encode 'other wins'
    # try one more heuristic: if there's a column suggesting 'OtherWin' (e.g., contains 'other' and 'win'),
    # invert that to produce FocalWin = 1 - OtherWin
    if df['FocalWin'].isna().all():
        for raw_col in original_raw_columns:
            low = raw_col.lower()
            if 'other' in low and 'win' in low and raw_col in df.columns:
                otherwin = pd.to_numeric(df[raw_col], errors='coerce')
                if not otherwin.isna().all():
                    df['FocalWin'] = 1.0 - otherwin.astype(float)
                    break

    # Final fallback: if FocalWin still all-NA but there exists a column named like 'winner' with text labels,
    # try to interpret 'focal'/'other' labels.
    if df['FocalWin'].isna().all():
        for raw_col in original_raw_columns:
            low = raw_col.lower()
            if ('winner' in low or 'winner' == low or 'result' in low or 'outcome' in low) and raw_col in df.columns:
                series = df[raw_col].astype(str).str.strip().str.lower()
                mapped = series.map(lambda s: 1.0 if 'focal' in s or 'win' == s or 'won' in s else (0.0 if 'other' in s or 'lose' in s or 'lost' in s else np.nan))
                if not mapped.isna().all():
                    df['FocalWin'] = pd.to_numeric(mapped, errors='coerce').astype(float)
                    break

    # Return the full dataframe with all required final columns present
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Ensure the model columns are present
    required = ['FocalWin', 'RelSize_z', 'DistanceDiff_z', 'Location', 'MaleAdvantage', 'DyadID']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f'Missing required columns for modeling: {missing}')

    # Do not attempt to fit on empty data
    if df.shape[0] == 0:
        raise ValueError('No rows available for modeling after transform.')

    # Work on a copy for modeling
    df_model = df.copy()

    # Ensure outcome is numeric and drop rows with missing outcome or core predictors
    df_model['FocalWin'] = pd.to_numeric(df_model['FocalWin'], errors='coerce')

    # Ensure Location is treated as a categorical variable (patsy/statsmodels will detect categories from data)
    df_model['Location'] = df_model['Location'].astype('category')

    # Fill any missing DyadID entries (should be none after transform, but be defensive)
    if df_model['DyadID'].isna().any():
        missing_mask = df_model['DyadID'].isna()
        df_model.loc[missing_mask, 'DyadID'] = [f'dyad_missing_{i}' for i in df_model[missing_mask].index]

    # Defensive fills for controls to avoid dropping rows unnecessarily
    if df_model['MaleAdvantage'].isna().any():
        df_model['MaleAdvantage'] = df_model['MaleAdvantage'].fillna(0.0)
    if df_model['Location'].isna().any():
        df_model['Location'] = df_model['Location'].fillna('Neutral')

    # For modeling, require non-missing outcome and the main predictors (RelSize_z, DistanceDiff_z).
    core_subset = ['FocalWin', 'RelSize_z', 'DistanceDiff_z', 'Location', 'MaleAdvantage', 'DyadID']
    df_model = df_model.dropna(subset=core_subset)

    if df_model.shape[0] == 0:
        raise ValueError('No complete cases available for modeling after dropping missing values in required columns.')

    # Formula: main effects of relative size and location (continuous & categorical), male advantage as control,
    # and an interaction testing whether the effect of relative size depends on location category.
    formula = 'FocalWin ~ RelSize_z + DistanceDiff_z + C(Location) + MaleAdvantage + RelSize_z:C(Location)'

    # Fit a logistic regression (binomial GLM)
    glm_model = smf.glm(formula=formula, data=df_model, family=sm.families.Binomial())
    results = glm_model.fit()

    # Obtain cluster-robust SEs by DyadID (accounts for non-independence of contests within dyads)
    try:
        clustered = results.get_robustcov_results(cov_type='cluster', groups=df_model['DyadID'])
    except Exception:
        # If clustering fails (e.g., too few clusters), fall back to the original results
        clustered = results

    # Return the fitted results object with robust covariances when available
    return clustered