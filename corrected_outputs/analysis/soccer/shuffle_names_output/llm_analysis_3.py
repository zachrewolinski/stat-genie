from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/shuffle_names_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to produce the columns used in modelling.

    Produces these final columns (exact names used in the model):
      - SkinAvg: average of rater1 and rater2 skin-tone ratings (0-1 normalized scale)
      - SkinCategory: categorical mapping into 'Light', 'Neutral', 'Dark'
      - SkinDark: binary indicator (1 = Dark, 0 = Light). We keep only clear 'Dark' and 'Light' cases.
      - RedCardCount: numeric count of red cards in the player-referee dyad
      - Matches: numeric count of matches the dyad played together (exposure)
      - RedCardRate: RedCardCount / Matches (descriptive rate)
      - ImplicitBias: country-level implicit bias score (leagueCountry column used)
      - ExplicitBias: country-level explicit bias score (club column used)
      - PlayerGoals: meanExp column used as player goals in dyad
      - PlayerYellowCards: playerShort column used as number of yellow cards
      - RefereeID: referee identifier (goals column used as referee id in this dataset schema)

    Notes on thresholds: rater1/rater2 were normalized to 0..1 on a 5-point scale (0, .25, .5, .75, 1).
    To focus on the research question contrast (dark vs light), we keep only those with average rating >= 0.75 as 'Dark' and <= 0.25 as 'Light'.

    """
    # Make a working copy
    df = df.copy()

    # Ensure numeric conversions where appropriate (coerce errors to NaN)
    numeric_cols = ['rater1', 'rater2', 'redCards', 'refNum', 'leagueCountry', 'club', 'meanExp', 'playerShort', 'goals']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Compute average skin rating from the two raters
    # (these are normalized 0..1; if a rater is missing we will get NaN and drop later)
    df['SkinAvg'] = df[['rater1', 'rater2']].mean(axis=1)

    # Create coarse skin category and binary dark vs light indicator
    # Thresholds chosen to capture clear 'light' (<=0.25) and 'dark' (>=0.75) ratings
    def skin_cat(x):
        if pd.isna(x):
            return pd.NA
        if x <= 0.25:
            return 'Light'
        if x >= 0.75:
            return 'Dark'
        return 'Neutral'

    df['SkinCategory'] = df['SkinAvg'].apply(skin_cat)
    # Binary indicator for dark (1) vs light (0)
    df['SkinDark'] = df['SkinCategory'].map({'Dark': 1, 'Light': 0})

    # Derive red card count and matches (exposure). Based on schema the column 'redCards' is used
    # and 'refNum' is treated here as the number of matches the player-referee dyad played (exposure).
    df['RedCardCount'] = df['redCards']
    df['Matches'] = df['refNum']

    # Remove rows with invalid/missing essential variables
    required = ['SkinDark', 'RedCardCount', 'Matches', 'leagueCountry', 'club', 'meanExp', 'playerShort', 'goals']
    # Keep only rows where SkinDark is not NA and is exactly 0 or 1 (i.e., Light or Dark)
    df = df[df['SkinDark'].notna()]

    # Convert to numeric and drop rows where matches is missing or zero (can't compute rate/offset)
    df = df[df['Matches'].notna()]
    df = df[df['Matches'] > 0]

    # Also drop rows with missing red card counts
    df = df[df['RedCardCount'].notna()]

    # Create descriptive rate column
    df['RedCardRate'] = df['RedCardCount'] / df['Matches']

    # Map control variables into clear names used in model
    df['ImplicitBias'] = df['leagueCountry']  # mean implicit bias score for referee country
    df['ExplicitBias'] = df['club']  # mean explicit bias score for referee country
    df['PlayerGoals'] = df['meanExp']  # player goals in dyad
    df['PlayerYellowCards'] = df['playerShort']  # yellow cards count from that referee

    # Referee identifier for clustering robust SEs
    # Schema indicates 'goals' column contains referee ID (anonymized); cast to integer where possible
    df['RefereeID'] = pd.to_numeric(df['goals'], errors='coerce').astype('Int64')

    # Drop rows missing any of the model covariates (we prefer a complete-case analysis for the primary model)
    model_covs = ['SkinDark', 'RedCardCount', 'Matches', 'ImplicitBias', 'ExplicitBias', 'PlayerGoals', 'PlayerYellowCards', 'RefereeID']
    df = df.dropna(subset=model_covs)

    # Ensure types: SkinDark is int, RefereeID is int
    df['SkinDark'] = df['SkinDark'].astype(int)
    # RefereeID may be nullable Int64; convert to plain int if possible
    df['RefereeID'] = df['RefereeID'].astype(int)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative-binomial generalized linear model for red-card counts with an offset for the number of matches.

    Model specification (primary):
      RedCardCount ~ SkinDark + ImplicitBias + ExplicitBias + PlayerGoals + PlayerYellowCards
    with offset = log(Matches).

    We use cluster-robust standard errors clustered by RefereeID to account for dependence within referees.

    Returns the fitted results object with cluster-robust covariance.
    """
    # Ensure required columns exist
    required = ['RedCardCount', 'Matches', 'SkinDark', 'ImplicitBias', 'ExplicitBias', 'PlayerGoals', 'PlayerYellowCards', 'RefereeID']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column {c} not found in dataframe")

    # Endogenous and exogenous
    y = df['RedCardCount'].astype(float)
    X = df[['SkinDark', 'ImplicitBias', 'ExplicitBias', 'PlayerGoals', 'PlayerYellowCards']].astype(float)
    X = sm.add_constant(X, has_constant='add')

    # Offset = log(number of matches)
    offset = np.log(df['Matches'].astype(float))

    # Fit a Negative Binomial GLM (handles overdispersion relative to Poisson)
    # Use statsmodels' GLM with NegativeBinomial family
    model_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
    res_nb = model_nb.fit()

    # Compute cluster-robust standard errors clustered by RefereeID
    # (this returns a results instance with adjusted covariance)
    try:
        res_nb_clust = res_nb.get_robustcov_results(cov_type='cluster', groups=df['RefereeID'])
    except Exception:
        # If clustering fails for any reason, fall back to the original result
        res_nb_clust = res_nb

    return res_nb_clust


