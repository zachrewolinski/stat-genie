from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/replace_with_rvs_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dyad dataframe into an analysis-ready dataframe.
    Produces:
      - skin_avg: mean of rater1 and rater2 (0-1 scale)
      - SkinCategory: 'dark' / 'light' / 'medium'
      - dark_skin: binary 1 if dark, 0 if light (rows restricted to dark or light only)
      - age_years: age at reference date (2013-01-01)
      - z-scored continuous controls: z_height, z_weight, z_age
      - goals_per_game, yellow_per_game
      - position flags: pos_DEF, pos_MID, pos_FWD, pos_GK, pos_Other
      - league flags: league_England, league_Germany, league_France, league_Spain
    """
    df = df.copy()

    # Drop rows missing essential variables
    df = df.dropna(subset=['rater1', 'rater2', 'redCards', 'games'])

    # Mean skin rating from two independent raters (0-1 scale)
    df['skin_avg'] = df[['rater1', 'rater2']].mean(axis=1)

    # Categorize into dark/light/medium for a focused comparison
    df['SkinCategory'] = df['skin_avg'].apply(lambda x: 'dark' if x >= 0.66 else ('light' if x <= 0.33 else 'medium'))

    # Restrict to dark and light players only (per the research question comparing dark vs light)
    df = df[df['SkinCategory'].isin(['dark', 'light'])].reset_index(drop=True)

    # Binary independent variable: 1 = dark, 0 = light
    df['dark_skin'] = (df['SkinCategory'] == 'dark').astype(int)

    # Parse birthday to compute age (birthday format is dd.mm.yyyy)
    df['birthday'] = pd.to_datetime(df['birthday'], dayfirst=True, errors='coerce')
    reference_date = pd.to_datetime('2013-01-01')
    df['age_years'] = (reference_date - df['birthday']).dt.days / 365.25

    # Ensure games is positive for offset use; drop rows with games <= 0
    df = df[df['games'] > 0].reset_index(drop=True)

    # Derived rates
    df['goals_per_game'] = df['goals'] / df['games']
    df['yellow_per_game'] = df['yellowCards'] / df['games']

    # Coarse mapping of position into broad role categories
    def map_position(pos):
        if pd.isnull(pos):
            return 'Other'
        p = str(pos).lower()
        if 'goal' in p or 'keeper' in p or 'gk' in p:
            return 'GK'
        if 'def' in p or 'back' in p or 'center back' in p or 'centre back' in p:
            return 'DEF'
        if 'mid' in p or 'wing' in p or 'half' in p:
            return 'MID'
        if 'forward' in p or 'striker' in p or 'attacking' in p or 'fw' in p or 'wing' in p:
            return 'FWD'
        return 'Other'

    df['pos_coarse'] = df['position'].apply(map_position)
    df['pos_DEF'] = (df['pos_coarse'] == 'DEF').astype(int)
    df['pos_MID'] = (df['pos_coarse'] == 'MID').astype(int)
    df['pos_FWD'] = (df['pos_coarse'] == 'FWD').astype(int)
    df['pos_GK'] = (df['pos_coarse'] == 'GK').astype(int)
    df['pos_Other'] = (df['pos_coarse'] == 'Other').astype(int)

    # League country flags (explicitly create the four league columns commonly present in dataset)
    df['league_England'] = (df['leagueCountry'] == 'England').astype(int)
    df['league_Germany'] = (df['leagueCountry'] == 'Germany').astype(int)
    df['league_France'] = (df['leagueCountry'] == 'France').astype(int)
    df['league_Spain'] = (df['leagueCountry'] == 'Spain').astype(int)

    # Standardize continuous covariates for numerical stability and interpretability
    for col in ['height', 'weight', 'age_years']:
        # compute mean and std on available values, avoid division by zero
        if col in df.columns:
            m = df[col].mean(skipna=True)
            s = df[col].std(skipna=True)
            if pd.isna(s) or s == 0:
                df['z_' + col.split('_')[0]] = 0.0
            else:
                df['z_' + col.split('_')[0]] = (df[col] - m) / s
        else:
            df['z_' + col.split('_')[0]] = 0.0

    # Keep relevant columns and return
    keep_cols = [
        'playerShort', 'player', 'club', 'leagueCountry', 'birthday', 'height', 'weight', 'position',
        'games', 'victories', 'ties', 'defeats', 'goals', 'yellowCards', 'yellowReds', 'redCards',
        'photoID', 'rater1', 'rater2', 'refNum', 'refCountry', 'meanIAT', 'nIAT', 'seIAT', 'meanExp', 'nExp', 'seExp',
        'skin_avg', 'SkinCategory', 'dark_skin', 'age_years', 'z_age', 'z_height', 'z_weight',
        'goals_per_game', 'yellow_per_game', 'pos_DEF', 'pos_MID', 'pos_FWD', 'pos_GK', 'pos_Other',
        'league_England', 'league_Germany', 'league_France', 'league_Spain'
    ]

    # Keep only columns that exist in df (safe subset)
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a Negative Binomial regression of redCards on dark_skin (primary IV),
    controlling for position, league, basic player covariates, and using games as an offset (exposure).
    Cluster standard errors by refNum (referee ID).

    Returns the fitted results object with clustered robust covariances.
    """
    # Ensure required columns exist
    required = ['redCards', 'games', 'dark_skin', 'z_height', 'z_weight', 'z_age',
                'goals_per_game', 'yellow_per_game', 'pos_DEF', 'pos_MID', 'pos_FWD', 'pos_GK', 'pos_Other',
                'league_England', 'league_Germany', 'league_France', 'league_Spain', 'meanIAT', 'meanExp', 'refNum']
    miss = [c for c in required if c not in df.columns]
    if len(miss) > 0:
        raise ValueError(f"Missing required columns for modeling: {miss}")

    # Select predictors
    predictors = [
        'dark_skin',
        'z_height', 'z_weight', 'z_age',
        'goals_per_game', 'yellow_per_game',
        'pos_DEF', 'pos_MID', 'pos_FWD', 'pos_GK', 'pos_Other',
        'league_England', 'league_Germany', 'league_France', 'league_Spain',
        'meanIAT', 'meanExp'
    ]

    # Prepare design matrix (fill NA with 0 for dummies/derived rates; continuous should already be numeric)
    X = df[predictors].fillna(0)
    X = sm.add_constant(X, has_constant='add')

    y = df['redCards'].astype(float)

    # offset = log(exposure games); ensures model predicts red card rate per game
    offset = np.log(df['games'].astype(float))

    # Fit Negative Binomial generalized linear model (handles overdispersion vs Poisson)
    model_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
    res_nb = model_nb.fit()

    # Obtain clustered robust covariance estimates by referee id
    try:
        res_nb_cluster = res_nb.get_robustcov_results(cov_type='cluster', groups=df['refNum'])
    except Exception:
        # Fallback: if clustering fails, return the plain result
        res_nb_cluster = res_nb

    # Print a brief summary (user can inspect returned results object)
    print(res_nb_cluster.summary())

    return res_nb_cluster


