from typing import Any
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Helper: case-insensitive column lookup
    def find_col_by_candidates(candidates):
        cols = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in cols:
                return cols[cand.lower()]
        return None

    # Helper: find columns matching regex or containing keywords (case-insensitive)
    def find_cols_by_pattern(patterns, min_count=1):
        found = []
        for col in df.columns:
            lname = col.lower()
            for p in patterns:
                if re.search(p, lname):
                    found.append(col)
                    break
        if len(found) >= min_count:
            return found
        return found

    # ===== RATER COLUMNS (rater1, rater2) =====
    # Try a variety of likely names for the two rater columns.
    rater1_candidates = ['rater1', 'rater_1', 'raterone', 'rater_one', 'rater.a', 'rater_a', 'skin_tone_1', 'skin_tone_rater1', 'skin1', 'raterA']
    rater2_candidates = ['rater2', 'rater_2', 'ratertwo', 'rater_two', 'rater.b', 'rater_b', 'skin_tone_2', 'skin_tone_rater2', 'skin2', 'raterB']

    r1_col = find_col_by_candidates(rater1_candidates)
    r2_col = find_col_by_candidates(rater2_candidates)

    # If we didn't find explicit 1/2 columns, look for any columns mentioning 'rater' or 'skin' or 'tone'
    if r1_col is None or r2_col is None:
        candidates = find_cols_by_pattern([r'rater', r'skin', r'tone'], min_count=2)
        # If we have 2+ candidates pick the top two distinct ones
        if len(candidates) >= 2:
            # choose two with the most non-null values
            candidates_sorted = sorted(candidates, key=lambda c: df[c].notna().sum(), reverse=True)
            if r1_col is None:
                r1_col = candidates_sorted[0]
            if r2_col is None:
                # ensure r2_col different from r1_col
                r2_col = next((c for c in candidates_sorted if c != r1_col), None)

    # If we still don't have rater columns, create them as NaN to preserve contract.
    if r1_col is None:
        df['rater1'] = np.nan
        r1_col = 'rater1'
    else:
        # copy/rename into canonical column name if necessary
        if r1_col != 'rater1':
            df['rater1'] = df[r1_col]

    if r2_col is None:
        df['rater2'] = np.nan
        r2_col = 'rater2'
    else:
        if r2_col != 'rater2':
            df['rater2'] = df[r2_col]

    # Coerce rater columns to numeric
    df['rater1'] = pd.to_numeric(df['rater1'], errors='coerce')
    df['rater2'] = pd.to_numeric(df['rater2'], errors='coerce')

    # If rater values appear to be on a 1-5 scale, normalize to [0,1] using (val - 1)/4.
    def normalize_rater(col):
        if df[col].notna().any():
            maxv = df[col].max(skipna=True)
            minv = df[col].min(skipna=True)
            # If max > 1.1 and values look like integers 1-5, map to 0-1
            if maxv is not None and maxv > 1.1:
                # attempt mapping for 1-5 scale
                df[col] = (df[col] - 1.0) / 4.0
            # If values are already in [0,1], leave as is.
    normalize_rater('rater1')
    normalize_rater('rater2')

    # Require both rater scores to be present (numeric) to compute SkinToneAvg.
    # If neither rater is present at all, we will end up filtering to zero rows;
    # that's legitimate only if no skin-tone information exists. But we try to avoid
    # dropping everything by proceeding only if there are any non-null ratings.
    if df['rater1'].notna().sum() == 0 and df['rater2'].notna().sum() == 0:
        # No rater info at all: create the required columns with NaN values so the contract is preserved.
        df['SkinToneAvg'] = np.nan
        df['SkinCategory'] = pd.Categorical([None] * len(df))
        df['DarkSkin'] = np.nan
    else:
        # If only one rater column has values, duplicate that rater into the other so we can compute an average.
        if df['rater1'].notna().sum() == 0 and df['rater2'].notna().sum() > 0:
            df['rater1'] = df['rater2']
        if df['rater2'].notna().sum() == 0 and df['rater1'].notna().sum() > 0:
            df['rater2'] = df['rater1']

        # Now drop rows that still lack either rater
        df = df[df['rater1'].notna() & df['rater2'].notna()].copy()

        # Average the two independent rater scores to get a continuous skin-tone measure
        df['SkinToneAvg'] = (df['rater1'] + df['rater2']) / 2.0

        # Create categorical skin tone groups so we can compare 'Dark' vs 'Light'.
        # Based on the normalized 5-point scale mapping into {0, .25, .5, .75, 1.0},
        # use thresholds: <=0.25 -> Light, >=0.75 -> Dark, middle values -> Medium.
        df['SkinCategory'] = pd.cut(df['SkinToneAvg'], bins=[-0.01, 0.25, 0.75, 1.01], labels=['Light', 'Medium', 'Dark'])

        # Keep only clear Light vs Dark cases for the primary comparison
        df = df[df['SkinCategory'].isin(['Light', 'Dark'])].copy()

        # Binary indicator for dark skin
        df['DarkSkin'] = (df['SkinCategory'] == 'Dark').astype(int)

    # If SkinToneAvg was set to NaN for all rows earlier, ensure DarkSkin column exists with NaN
    if 'DarkSkin' not in df.columns:
        df['DarkSkin'] = np.nan

    # ===== DEPENDENT VARIABLE: red_card_count =====
    red_count_sources = [
        'red_card_count', 'red_cards', 'redcards', 'redCards', 'redCardCount', 'redcard', 'red_card', 'redCard', 'photoID'
    ]
    red_assigned = False
    for src in red_count_sources:
        # case-insensitive match
        match = find_col_by_candidates([src])
        if match is not None:
            df['red_card_count'] = pd.to_numeric(df[match], errors='coerce')
            red_assigned = True
            break
    if not red_assigned:
        # attempt to find columns that look like counts of red cards
        candidates = find_cols_by_pattern([r'red', r'card'], min_count=1)
        if candidates:
            # pick the most promising by numeric content
            c = max(candidates, key=lambda col: df[col].notna().sum())
            df['red_card_count'] = pd.to_numeric(df[c], errors='coerce')
            red_assigned = True
    if not red_assigned:
        # create the column to satisfy the contract; leave values as NaN
        df['red_card_count'] = np.nan

    # ===== EXPOSURE: ExposureMatches =====
    exposure_sources = ['ExposureMatches', 'matches', 'num_matches', 'exposure', 'games', 'numMatches', 'num_matches']
    exposure_assigned = False
    for src in exposure_sources:
        match = find_col_by_candidates([src])
        if match is not None:
            df['ExposureMatches'] = pd.to_numeric(df[match], errors='coerce')
            exposure_assigned = True
            break
    if not exposure_assigned:
        # try to find columns that look like match counts
        candidates = find_cols_by_pattern([r'match', r'game', r'exposure'], min_count=1)
        if candidates:
            c = max(candidates, key=lambda col: df[col].notna().sum())
            df['ExposureMatches'] = pd.to_numeric(df[c], errors='coerce')
            exposure_assigned = True
    if not exposure_assigned:
        df['ExposureMatches'] = np.nan

    # If exposure has zero or negative values or is missing for rows where red_card_count is present,
    # replace with 1 as a conservative default to allow modeling (avoids log(0)). This is a pragmatic
    # imputation to keep the pipeline runnable; it preserves rates when counts are zero.
    if 'ExposureMatches' in df.columns:
        # For rows where ExposureMatches is missing but red_card_count is present, set ExposureMatches to 1
        mask_fix = df['ExposureMatches'].isna() & df['red_card_count'].notna()
        if mask_fix.any():
            df.loc[mask_fix, 'ExposureMatches'] = 1.0
        # For any non-positive exposures, set to 1 to avoid log(0) issues
        mask_nonpos = df['ExposureMatches'].notna() & (df['ExposureMatches'] <= 0)
        if mask_nonpos.any():
            df.loc[mask_nonpos, 'ExposureMatches'] = 1.0

    # ===== CONTROLS =====
    # meanIAT
    meanIAT_sources = ['meanIAT', 'mean_iat', 'iat_mean', 'meanIat']
    meanIAT_assigned = False
    for src in meanIAT_sources:
        match = find_col_by_candidates([src])
        if match is not None:
            df['meanIAT'] = pd.to_numeric(df[match], errors='coerce')
            meanIAT_assigned = True
            break
    if not meanIAT_assigned:
        candidates = find_cols_by_pattern([r'iat', r'implicit'], min_count=1)
        if candidates:
            c = max(candidates, key=lambda col: df[col].notna().sum())
            df['meanIAT'] = pd.to_numeric(df[c], errors='coerce')
            meanIAT_assigned = True
    if not meanIAT_assigned:
        df['meanIAT'] = np.nan

    # club (explicit bias / thermometer)
    club_sources = ['club', 'explicitBias', 'thermometer', 'club_score']
    club_assigned = False
    for src in club_sources:
        match = find_col_by_candidates([src])
        if match is not None:
            df['club'] = pd.to_numeric(df[match], errors='coerce')
            club_assigned = True
            break
    if not club_assigned:
        candidates = find_cols_by_pattern([r'therm', r'explic', r'club'], min_count=1)
        if candidates:
            c = max(candidates, key=lambda col: df[col].notna().sum())
            df['club'] = pd.to_numeric(df[c], errors='coerce')
            club_assigned = True
    if not club_assigned:
        df['club'] = np.nan

    # playerShort (number of yellow cards between player-referee dyad)
    playerShort_sources = ['playerShort', 'player_short', 'yellow_cards', 'yellowCards', 'yellows', 'playerShortCount']
    playerShort_assigned = False
    for src in playerShort_sources:
        match = find_col_by_candidates([src])
        if match is not None:
            df['playerShort'] = pd.to_numeric(df[match], errors='coerce')
            playerShort_assigned = True
            break
    if not playerShort_assigned:
        candidates = find_cols_by_pattern([r'yellow', r'yellows', r'yellow_card', r'player'], min_count=1)
        if candidates:
            c = max(candidates, key=lambda col: df[col].notna().sum())
            df['playerShort'] = pd.to_numeric(df[c], errors='coerce')
            playerShort_assigned = True
    if not playerShort_assigned:
        df['playerShort'] = np.nan

    # victories (league country / fixed effect). If missing, set to 'Unknown'
    victories_sources = ['victories', 'league', 'league_country', 'country']
    victories_assigned = False
    for src in victories_sources:
        match = find_col_by_candidates([src])
        if match is not None:
            df['victories'] = df[match].fillna('Unknown')
            victories_assigned = True
            break
    if not victories_assigned:
        df['victories'] = 'Unknown'

    # Impute control missing values with medians (for numeric controls) to avoid dropping many rows in modeling.
    # This is a pragmatic choice to keep pipeline runnable; if a control has no observed values, fill with 0.
    for col in ['meanIAT', 'club', 'playerShort']:
        if col in df.columns:
            if df[col].notna().sum() > 0:
                median_val = df[col].median(skipna=True)
                df[col] = df[col].fillna(median_val)
            else:
                df[col] = df[col].fillna(0.0)

    # Ensure DarkSkin column is integer for rows where it's known
    if 'DarkSkin' in df.columns:
        # If values are boolean/float, coerce to numeric ints where possible, keep NaN otherwise
        df['DarkSkin'] = pd.to_numeric(df['DarkSkin'], errors='coerce')

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Negative binomial regression for count outcome with exposure (number of matches)
    # Uses a log-offset of ExposureMatches so coefficients represent rate ratios per match.

    # Work on a copy
    df = df.copy()

    # Basic sanity checks: ensure required columns exist in the dataframe
    required_columns = ['DarkSkin', 'red_card_count', 'ExposureMatches', 'meanIAT', 'club', 'playerShort', 'victories']
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"The following required columns are missing from the dataframe: {missing}")

    if df.shape[0] == 0:
        raise ValueError("The transformed dataframe is empty; cannot fit model.")

    # Ensure there is at least one non-missing value for 'victories' (categorical)
    if df['victories'].notna().sum() == 0:
        df['victories'] = df['victories'].fillna('Unknown')

    # Build a clean dataset for modeling by selecting variables
    model_vars = ['red_card_count', 'DarkSkin', 'meanIAT', 'club', 'playerShort', 'victories', 'ExposureMatches']
    df_model = df[model_vars].copy()

    # Require non-missing values for the core variables: DV, IV, Exposure
    core_required = ['red_card_count', 'DarkSkin', 'ExposureMatches']
    df_model = df_model.dropna(subset=core_required)

    if df_model.shape[0] == 0:
        raise ValueError("No observations with complete data for the core model variables (red_card_count, DarkSkin, ExposureMatches).")

    # For any remaining missing controls (should be few due to imputation in transform), fill with median/0
    for col in ['meanIAT', 'club', 'playerShort']:
        if df_model[col].notna().sum() == 0:
            df_model[col] = 0.0
        else:
            df_model[col] = df_model[col].fillna(df_model[col].median(skipna=True))

    # Ensure exposure is strictly positive for all remaining observations
    if (df_model['ExposureMatches'] <= 0).any():
        raise ValueError("ExposureMatches must be strictly positive for all observations used in the model to compute the log-offset.")

    # Primary model specification: effect of DarkSkin on red card rate, controlling for
    # meanIAT (implicit bias), club (explicit bias), playerShort (yellow cards from same referee),
    # and league fixed effects (victories). Exposure is number of matches (ExposureMatches).
    formula = 'red_card_count ~ DarkSkin + meanIAT + club + playerShort + C(victories)'

    # Fit the model using glm Negative Binomial with log offset
    fitted_model = smf.glm(formula=formula,
                          data=df_model,
                          family=sm.families.NegativeBinomial(),
                          offset=np.log(df_model['ExposureMatches'])).fit()

    return fitted_model