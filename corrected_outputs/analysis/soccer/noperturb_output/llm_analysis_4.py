from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/noperturb_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a working copy
    df = df.copy()

    # Drop rows with missing outcome or exposure
    df = df.dropna(subset=['redCards', 'games'])
    # keep only dyads with at least one game (exposure must be > 0 for offset)
    df = df[df['games'] > 0]

    # Rater scores: ensure numeric and drop rows missing raters
    df = df.dropna(subset=['rater1', 'rater2'])
    df['rater1'] = pd.to_numeric(df['rater1'], errors='coerce')
    df['rater2'] = pd.to_numeric(df['rater2'], errors='coerce')
    df = df.dropna(subset=['rater1', 'rater2'])

    # Create average skin score (range expected 0.0 - 1.0 given dataset normalization)
    df['SkinScore'] = (df['rater1'] + df['rater2']) / 2.0

    # Create categorical bins: focus the main comparison on extreme categories
    # The original ratings are on a 5-point scale normalized to [0,1] (0,0.25,0.5,0.75,1.0)
    # Define 'light' as <= 0.25 and 'dark' as >= 0.75; exclude middle (0.5) to sharpen contrast
    def _skin_bin(s):
        if pd.isna(s):
            return pd.NA
        if s <= 0.25:
            return 'light'
        if s >= 0.75:
            return 'dark'
        return 'middle'

    df['SkinBin'] = df['SkinScore'].apply(_skin_bin)

    # Keep only light and dark categories (exclude middle/ambiguous)
    df = df[df['SkinBin'].isin(['light', 'dark'])].copy()

    # Binary indicator for dark skin
    df['SkinDark'] = (df['SkinBin'] == 'dark').astype(int)

    # Convert birthday to datetime and compute age at reference date (2013-01-01)
    # Some birthdays may be in format 'dd.mm.yyyy'. Use errors='coerce' to handle anomalies.
    df['birthday'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y', errors='coerce')
    ref_date = pd.to_datetime('2013-01-01')

    def compute_age(birth):
        if pd.isna(birth):
            return np.nan
        age = ref_date.year - birth.year - ((ref_date.month, ref_date.day) < (birth.month, birth.day))
        return age

    df['Age'] = df['birthday'].apply(compute_age)

    # Log of games for offset (already filtered to games > 0)
    df['log_games'] = np.log(df['games'])

    # Ensure numeric controls are numeric
    df['height'] = pd.to_numeric(df['height'], errors='coerce')
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df['meanIAT'] = pd.to_numeric(df['meanIAT'], errors='coerce')
    df['meanExp'] = pd.to_numeric(df['meanExp'], errors='coerce')

    # Keep columns needed for modeling and diagnostics
    # (retain position, leagueCountry, playerShort for dummies/clustering)
    keep_cols = [
        'playerShort', 'player', 'refNum', 'refCountry', 'leagueCountry', 'position',
        'games', 'log_games', 'redCards', 'SkinScore', 'SkinBin', 'SkinDark',
        'Age', 'height', 'weight', 'meanIAT', 'meanExp'
    ]
    # If some of these columns are missing from original df, keep only those present
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    # Final: drop rows with missing required numeric controls (Age, height, weight, meanIAT/meanExp are optional but good to have)
    # We'll not drop rows just because a control is missing; model will handle NaNs (but statsmodels requires no NaNs in X).
    # However, drop rows missing the primary variables (redCards, games, SkinDark)
    df = df.dropna(subset=['redCards', 'games', 'SkinDark'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Build design matrix and fit a negative binomial model for count outcome with games as exposure
    # Assumes df already transformed by transform() above
    data = df.copy()

    # Required columns check
    required = ['redCards', 'log_games', 'SkinDark']
    for c in required:
        if c not in data.columns:
            raise ValueError(f"Required column {c} not found in dataframe")

    # Create dummy variables for categorical controls (drop first to avoid multicollinearity)
    # Position dummies
    if 'position' in data.columns:
        pos_dummies = pd.get_dummies(data['position'].astype(str), prefix='pos', drop_first=True)
    else:
        pos_dummies = pd.DataFrame(index=data.index)

    # League country dummies
    if 'leagueCountry' in data.columns:
        league_dummies = pd.get_dummies(data['leagueCountry'].astype(str), prefix='league', drop_first=True)
    else:
        league_dummies = pd.DataFrame(index=data.index)

    # Base covariates
    base_covs = ['SkinDark']
    # Add continuous controls if present
    for col in ['meanIAT', 'meanExp', 'height', 'weight', 'Age']:
        if col in data.columns:
            base_covs.append(col)

    # Construct X and y
    X = pd.DataFrame(index=data.index)
    for col in base_covs:
        if col in data.columns:
            X[col] = data[col]
    # add dummies
    X = pd.concat([X, pos_dummies, league_dummies], axis=1)

    # Drop rows with any NaNs in X or y because statsmodels requires complete cases
    y = data['redCards']
    complete_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[complete_idx]
    y = y.loc[complete_idx]
    offset = data.loc[complete_idx, 'log_games']

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Determine clustering variable if available
    cluster_var = 'playerShort' if 'playerShort' in data.columns else ('refNum' if 'refNum' in data.columns else None)
    clusters = data.loc[complete_idx, cluster_var] if cluster_var is not None else None

    # Fit negative binomial GLM with log link and offset = log_games
    # This models E[redCards] = exp(X beta + offset) where offset = log(games)
    try:
        if clusters is not None:
            res = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset).fit(
                cov_type='cluster', cov_kwds={'groups': clusters}
            )
        else:
            res = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset).fit(cov_type='HC1')
    except Exception:
        # Fallback to Poisson if NegativeBinomial fails
        if clusters is not None:
            res = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit(
                cov_type='cluster', cov_kwds={'groups': clusters}
            )
        else:
            res = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit(cov_type='HC1')

    # Return the fitted results object (it will reflect robust cov_type if requested and has .summary())
    return res