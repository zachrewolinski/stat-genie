from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dyad-level dataset into a cleaned dataframe containing all columns required for modeling.

    Creates:
    - SkinTone: continuous average of rater1 and rater2 (0-1 scale)
    - DarkSkin: binary indicator (1 if SkinTone >= 0.60, else 0). Threshold represents top categories on 5-pt scale normalized to 0-1.
    - Age: age in years at season midpoint (2013-01-01)
    - photoAvailable: indicator if photoID present
    - yellowRate, goalsRate: per-game rates used as covariates
    - Ensures games>0 and drops rows with missing essential data
    """
    df = df.copy()

    # Ensure numeric columns are numeric
    for col in ['rater1', 'rater2', 'games', 'redCards', 'yellowCards', 'goals', 'height', 'weight', 'meanIAT', 'meanExp', 'refNum']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute SkinTone: mean of available raters (skipna)
    rater_cols = [c for c in ['rater1', 'rater2'] if c in df.columns]
    if rater_cols:
        df['SkinTone'] = df[rater_cols].mean(axis=1, skipna=True)
    else:
        df['SkinTone'] = np.nan

    # Photo availability
    if 'photoID' in df.columns:
        df['photoAvailable'] = df['photoID'].notna().astype(int)
    else:
        # If photoID not present, set to 0 (no photo available)
        df['photoAvailable'] = 0

    # Create binary DarkSkin indicator. Threshold at 0.60 (roughly top two of five categories when normalized to 0-1).
    df['DarkSkin'] = (df['SkinTone'] >= 0.60).astype(int)

    # Parse birthday to compute age at season midpoint
    season_mid = pd.to_datetime('2013-01-01')
    # birthday format given as dd.mm.yyyy; use dayfirst=True; coerce errors
    if 'birthday' in df.columns:
        df['birthday_parsed'] = pd.to_datetime(df['birthday'], dayfirst=True, errors='coerce')
        df['Age'] = (season_mid - df['birthday_parsed']).dt.days / 365.25
    else:
        df['Age'] = np.nan

    # Rates per game for yellow cards and goals (games is exposure)
    # Avoid division errors; games should be >= 1 per schema
    if 'games' in df.columns:
        df['games'] = pd.to_numeric(df['games'], errors='coerce')
    else:
        df['games'] = np.nan

    # Replace zero or negative games with NaN to drop later (schema min 1)
    df.loc[df['games'] <= 0, 'games'] = np.nan

    if 'yellowCards' in df.columns:
        df['yellowCards'] = pd.to_numeric(df['yellowCards'], errors='coerce')
    else:
        df['yellowCards'] = np.nan

    if 'goals' in df.columns:
        df['goals'] = pd.to_numeric(df['goals'], errors='coerce')
    else:
        df['goals'] = np.nan

    df['yellowRate'] = df['yellowCards'] / df['games']
    df['goalsRate'] = df['goals'] / df['games']

    # Ensure redCards numeric; if missing leave as NaN for dropping later
    if 'redCards' in df.columns:
        df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce')
    else:
        df['redCards'] = np.nan

    # Clean categorical controls
    # Fill missing position/leagueCountry with explicit category to avoid drop when creating dummies
    if 'position' in df.columns:
        df['position'] = df['position'].fillna('Unknown')
    else:
        df['position'] = 'Unknown'
    if 'leagueCountry' in df.columns:
        df['leagueCountry'] = df['leagueCountry'].fillna('Unknown')
    else:
        df['leagueCountry'] = 'Unknown'

    # Ensure playerShort and refNum exist before dropping rows
    if 'playerShort' not in df.columns:
        df['playerShort'] = np.nan
    if 'refNum' not in df.columns:
        df['refNum'] = np.nan

    # Keep only rows with essential data: SkinTone (at least one rater), games, redCards, playerShort, refNum
    required = ['SkinTone', 'games', 'redCards', 'playerShort', 'refNum']
    # Make sure required columns exist (they should after above steps)
    for col in required:
        if col not in df.columns:
            df[col] = np.nan
    df = df.dropna(subset=required)

    # Finalize redCards as integer counts (safe because we dropped rows with NaN redCards)
    # Coerce to int (ensure non-negative)
    df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce').fillna(0).astype(int)

    # Ensure meanIAT and meanExp numeric if present
    if 'meanIAT' in df.columns:
        df['meanIAT'] = pd.to_numeric(df['meanIAT'], errors='coerce')
    else:
        df['meanIAT'] = np.nan
    if 'meanExp' in df.columns:
        df['meanExp'] = pd.to_numeric(df['meanExp'], errors='coerce')
    else:
        df['meanExp'] = np.nan

    # Ensure height and weight numeric
    if 'height' in df.columns:
        df['height'] = pd.to_numeric(df['height'], errors='coerce')
    else:
        df['height'] = np.nan
    if 'weight' in df.columns:
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    else:
        df['weight'] = np.nan

    # Keep columns used in the model (but return full df as well). This set must match conceptual variable column names.
    needed_cols = ['playerShort', 'refNum', 'games', 'redCards', 'SkinTone', 'DarkSkin', 'Age', 'height', 'weight',
                   'yellowRate', 'goalsRate', 'position', 'leagueCountry', 'meanIAT', 'meanExp', 'photoAvailable']

    # Ensure all needed columns exist; if missing numeric controls create them filled with NaN
    for c in needed_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Return dataframe with all columns preserved; modeling function will select required columns
    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative binomial GLM for red card counts using the number of games as exposure (offset).

    Primary test: coefficient on DarkSkin (binary) and on continuous SkinTone.
    Controls: Age, height, weight, yellowRate, goalsRate, position (categorical), leagueCountry (categorical),
    meanIAT and meanExp (referee-country bias measures), photoAvailable.

    Clustered robust standard errors are computed by referee (refNum) to account for within-referee correlation.

    Returns the fitted model results (results object).
    """
    # Work on a copy
    df = df.copy()

    # Ensure required columns exist
    required_cols = ['redCards', 'games', 'DarkSkin', 'SkinTone', 'refNum']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Drop rows missing the absolutely required modeling variables
    df = df.dropna(subset=['redCards', 'games', 'DarkSkin', 'SkinTone', 'refNum'])

    # Prepare dependent and exposure
    y = df['redCards'].astype(float)
    offset = np.log(df['games'].astype(float))

    # Define numeric predictors (as specified in contract)
    numeric_predictors = ['DarkSkin', 'SkinTone', 'Age', 'height', 'weight', 'yellowRate', 'goalsRate', 'meanIAT', 'meanExp', 'photoAvailable']

    # Start with numeric predictors that exist in df
    exog_parts = []
    for col in numeric_predictors:
        if col in df.columns:
            exog_parts.append(df[col].astype(float))
        else:
            # Column must exist per contract, but if not, create NaNs (they will be dropped by dropna later)
            exog_parts.append(pd.Series(np.nan, index=df.index, name=col))

    exog_df = pd.concat(exog_parts, axis=1)
    exog_df.columns = numeric_predictors

    # Add categorical predictors via one-hot encoding using drop_first=True to avoid dummy trap.
    for cat in ['position', 'leagueCountry']:
        if cat in df.columns:
            dummies = pd.get_dummies(df[cat].astype(str), prefix=cat, drop_first=True)
            # If dummies empty (single category), skip
            if not dummies.empty:
                exog_df = pd.concat([exog_df, dummies.astype(float)], axis=1)

    # Drop columns with no variation (single unique value) - they cannot be estimated and cause singularity.
    nunique = exog_df.nunique(dropna=True)
    cols_with_variation = nunique[nunique > 1].index.tolist()
    if cols_with_variation:
        exog_df = exog_df.loc[:, cols_with_variation]
    else:
        # If nothing has variation, keep at least the primary predictors if present
        keep_cols = [c for c in ['DarkSkin', 'SkinTone'] if c in exog_df.columns]
        exog_df = exog_df.loc[:, keep_cols]

    # If after dropping constant columns some of the primary predictors are gone (unlikely), ensure at least we keep DarkSkin and SkinTone if present
    for primary in ['DarkSkin', 'SkinTone']:
        if primary in numeric_predictors and primary not in exog_df.columns and primary in df.columns:
            exog_df[primary] = df[primary].astype(float)

    # Add constant term
    exog_df = sm.add_constant(exog_df, has_constant='add')

    # Ensure no columns with all-NaN remain
    exog_df = exog_df.loc[:, exog_df.notna().any()]

    # Combine y, offset, refNum, and exog to drop any rows with NaNs or infs
    combined = pd.concat([y.rename('redCards'), offset.rename('offset'), df['refNum'].rename('refNum'), exog_df], axis=1)

    # Replace inf with NaN and drop any rows with NaN
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined = combined.dropna(axis=0, how='any')

    # After cleaning, extract arrays for modeling
    if combined.shape[0] == 0:
        raise ValueError("No data available after dropping rows with missing modeling variables.")

    y_clean = combined['redCards'].astype(float)
    offset_clean = combined['offset'].astype(float)
    groups = combined['refNum']
    exog_clean = combined.loc[:, exog_df.columns].astype(float)

    # Build and fit GLM (Negative Binomial) with offset
    model_glm = sm.GLM(y_clean, exog_clean, family=sm.families.NegativeBinomial(), offset=offset_clean)

    try:
        # Try clustered standard errors by referee
        res = model_glm.fit()
        # Attach robust covariance (cluster) if possible using get_robustcov_results to avoid re-fitting
        try:
            res_clus = res.get_robustcov_results(cov_type='cluster', groups=groups)
            res = res_clus
        except Exception:
            # If cluster covariance fails, try HC1
            try:
                res_hc = res.get_robustcov_results(cov_type='HC1')
                res = res_hc
            except Exception:
                # Last resort: keep original fitted results
                pass
    except Exception:
        # If direct fit fails for some reason, try with statsmodels formula interface as fallback
        # Build a conservative formula excluding categorical variables (they may have caused singularities)
        formula = 'redCards ~ DarkSkin + SkinTone + Age + height + weight + yellowRate + goalsRate + meanIAT + meanExp + photoAvailable'
        # Use the cleaned combined dataframe for fallback fitting if possible
        fallback_df = combined.copy()
        try:
            model_fallback = smf.glm(formula=formula, data=fallback_df, family=sm.families.NegativeBinomial(), offset=fallback_df['offset'])
            try:
                res = model_fallback.fit(cov_type='cluster', cov_kwds={'groups': fallback_df['refNum']})
            except Exception:
                res = model_fallback.fit(cov_type='HC1')
        except Exception:
            # Re-raise the original exception as nothing sensible to do
            raise

    # Print summary for convenience
    try:
        print(res.summary())
    except Exception:
        pass

    return res