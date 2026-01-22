from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # --- Basic filtering: need rater scores, games > 0, and redCards (count) ---
    df = df.dropna(subset=['rater1', 'rater2', 'games', 'redCards'])
    df = df[df['games'] > 0]

    # --- Derive mean skin rating from two raters ---
    df['skin_mean'] = df[['rater1', 'rater2']].mean(axis=1)

    # Define thresholds for 'light' and 'dark' to compare clearly distinct groups.
    low_thresh = 0.4   # corresponds to roughly the lower/lighter end of the 5-point scale
    high_thresh = 0.6  # corresponds to roughly the darker end

    df['is_dark'] = (df['skin_mean'] >= high_thresh).astype(int)
    df['is_light'] = (df['skin_mean'] <= low_thresh).astype(int)

    # Keep only clearly dark or light observations to focus the comparison
    df = df[(df['is_dark'] == 1) | (df['is_light'] == 1)].copy()

    # --- Parse birthday and compute age at reference date (season midpoint/standard date) ---
    # Birthdays are in dd.mm.yyyy format per schema
    df['birthday'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y', errors='coerce')
    season_ref = pd.to_datetime('2013-01-01')
    df['age'] = (season_ref - df['birthday']).dt.days / 365.25

    # --- Ensure control numeric columns exist and impute median where necessary ---
    numeric_controls = ['height', 'weight', 'yellowCards', 'yellowReds', 'goals', 'meanIAT', 'meanExp']
    for col in numeric_controls:
        if col in df.columns:
            # replace obvious missing with median to retain observations
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            # create column of zeros if not present (should not happen given schema)
            df[col] = 0

    # For categorical controls, ensure consistent dtypes
    if 'position' in df.columns:
        df['position'] = df['position'].astype('category')
    else:
        df['position'] = pd.Categorical(['Unknown'] * len(df))

    if 'leagueCountry' in df.columns:
        df['leagueCountry'] = df['leagueCountry'].astype('category')
    else:
        df['leagueCountry'] = pd.Categorical(['Unknown'] * len(df))

    # Convert refNum to integer for clustering later
    if 'refNum' in df.columns:
        df['refNum'] = pd.to_numeric(df['refNum'], errors='coerce')
        df = df.dropna(subset=['refNum'])
        df['refNum'] = df['refNum'].astype(int)
    else:
        # If refNum missing, create a placeholder (will prevent clustering but keep column present)
        df['refNum'] = pd.Series([0] * len(df), dtype=int)

    # Ensure games and redCards are numeric
    df['games'] = pd.to_numeric(df['games'], errors='coerce')
    df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce')

    # Filter out any remaining invalid rows
    df = df.dropna(subset=['games', 'redCards', 'skin_mean', 'is_dark'])

    # Create log of games for offset in count model
    df['log_games'] = np.log(df['games'].astype(float))

    # Final dataframe returned contains all columns needed for the model
    wanted_cols = [
        'playerShort', 'player', 'club', 'leagueCountry', 'position', 'birthday',
        'age', 'height', 'weight', 'games', 'log_games', 'redCards',
        'yellowCards', 'yellowReds', 'goals', 'skin_mean', 'is_dark', 'is_light',
        'meanIAT', 'meanExp', 'refNum'
    ]

    # Keep only columns that exist in df (safe subset)
    wanted_cols = [c for c in wanted_cols if c in df.columns]
    df = df[wanted_cols].reset_index(drop=True)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative binomial regression of red-card counts on darker-skin indicator
    controlling for player and country covariates. Use games as the exposure (offset)
    and cluster SEs by referee (refNum).

    Returns: statsmodels results object with clustered robust covariances (if possible).
    """
    # Ensure the offset is present; if not, compute
    if 'log_games' not in df.columns:
        df['log_games'] = np.log(df['games'].astype(float))

    # Formula: redCards as a function of is_dark plus controls; categorical variables wrapped with C()
    formula = (
        'redCards ~ is_dark + age + height + weight + yellowCards + yellowReds + goals '
        '+ meanIAT + meanExp + C(position) + C(leagueCountry)'
    )

    # Fit negative binomial GLM with log link and offset = log(games)
    try:
        glm_nb = smf.glm(formula=formula, data=df,
                         family=sm.families.NegativeBinomial(),
                         offset=df['log_games']).fit()
    except Exception:
        # If negative binomial family fails for some reason, fallback to Poisson with robust SE
        glm_nb = smf.glm(formula=formula, data=df,
                         family=sm.families.Poisson(),
                         offset=df['log_games']).fit()

    # Obtain cluster-robust standard errors clustered by referee (refNum)
    # If refNum is not available or clustering fails, fall back to the original fitted model
    res_cluster = glm_nb
    if 'refNum' in df.columns:
        try:
            res_cluster = glm_nb.get_robustcov_results(cov_type='cluster', groups=df['refNum'])
        except Exception:
            res_cluster = glm_nb

    return res_cluster