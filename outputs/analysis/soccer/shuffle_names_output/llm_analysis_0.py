from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure relevant columns are present and coerce to numeric where helpful
    # Columns used: 'rater1', 'rater2', 'redCards', 'games', 'leagueCountry', 'club', 'photoID', 'playerShort', 'goals'
    for col in ['rater1', 'rater2', 'redCards', 'games', 'leagueCountry', 'club', 'photoID', 'playerShort', 'goals']:
        if col in df.columns:
            # Try to coerce to numeric where appropriate
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # If any required column is missing, create it as NaN so downstream code fails clearly
            df[col] = np.nan

    # Drop rows with missing key outcome/exposure or rater info
    df = df.dropna(subset=['redCards', 'games', 'rater1', 'rater2'])

    # Keep only dyads where there was at least one game (exposure > 0)
    df = df[df['games'] > 0]

    # Compute average skin tone from two raters (scale normalized 0..1 where 0 ~ very light, 1 ~ very dark)
    df['skin_avg'] = df[['rater1', 'rater2']].mean(axis=1)

    # Define extreme groups: treat <=0.25 as 'Light', >=0.75 as 'Dark'. Remove middle category to maximize contrast for the research question.
    df['SkinGroup'] = np.where(df['skin_avg'] >= 0.75, 'Dark', np.where(df['skin_avg'] <= 0.25, 'Light', 'Middle'))
    df = df[df['SkinGroup'].isin(['Dark', 'Light'])].copy()

    # Binary indicator used in models (1 = Dark, 0 = Light)
    df['IsDark'] = (df['SkinGroup'] == 'Dark').astype(int)

    # Keep integer counts for the DV
    df['redCards'] = df['redCards'].astype(int)

    # Standardize continuous country-level bias measures for easier coefficient interpretation
    # Use leagueCountry (implicit) and club (explicit) per dataset description
    lc_mean = df['leagueCountry'].mean()
    lc_std = df['leagueCountry'].std(ddof=0)
    df['leagueCountry_z'] = (df['leagueCountry'] - lc_mean) / (lc_std if lc_std != 0 else 1)

    club_mean = df['club'].mean()
    club_std = df['club'].std(ddof=0)
    df['club_z'] = (df['club'] - club_mean) / (club_std if club_std != 0 else 1)

    # Ensure photoID and playerShort are numeric and fillna with 0 where sensible
    df['photoID'] = pd.to_numeric(df['photoID'], errors='coerce').fillna(0).astype(int)
    df['playerShort'] = pd.to_numeric(df['playerShort'], errors='coerce').fillna(0).astype(int)

    # Ensure goals (referee id for clustering) is present and coerced to int (keep original values if numeric)
    df['goals'] = pd.to_numeric(df['goals'], errors='coerce').fillna(-1).astype(int)

    # Keep only columns necessary for the statistical model and diagnostics
    keep_cols = ['IsDark', 'redCards', 'games', 'leagueCountry_z', 'club_z', 'photoID', 'playerShort', 'goals', 'skin_avg', 'SkinGroup']
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a Negative Binomial GLM for red card counts with exposure (number of games).
    Returns the model fit and a version with referee-clustered robust SEs.
    """
    # Prepare data for modeling
    df = df.copy()

    # Outcome and offset (exposure)
    y = df['redCards']
    offset = np.log(df['games'])

    # Predictors: intercept + IsDark + controls
    X = df[['IsDark', 'leagueCountry_z', 'club_z', 'photoID', 'playerShort']].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # Fit Negative Binomial (GLM) with log link and offset = log(games)
    model_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)

    # Fit raw model
    res_nb = model_nb.fit()

    # Obtain referee-clustered robust covariance (cluster by referee id stored in 'goals')
    groups = df['goals'].fillna(-1).astype(int)

    # Attempt to fit model with clustered covariance via fit(cov_type=...)
    try:
        # Many statsmodels versions support specifying cov_type and cov_kwds in fit
        res_nb_cluster = model_nb.fit(cov_type='cluster', cov_kwds={'groups': groups})
    except Exception:
        # Fallback to HC1 robust cov if clustering via fit fails
        try:
            res_nb_cluster = model_nb.fit(cov_type='HC1')
        except Exception:
            # As a last resort, return the raw model for both entries
            res_nb_cluster = res_nb

    # Return both the raw NB fit and the clustered-robust results object
    return {
        'nb_model_raw': res_nb,
        'nb_model_clustered_se': res_nb_cluster
    }