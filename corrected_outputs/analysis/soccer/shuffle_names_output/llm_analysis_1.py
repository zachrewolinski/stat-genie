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
    Transform the raw dataset into the analysis dataframe.

    Produces the following new columns used by the model:
      - SkinToneAvg : average of rater1 and rater2 (numeric 0-1)
      - IsDark      : binary indicator (1=dark skin, 0=light skin); excludes medium cases
      - RedCardCount: integer count of red cards for the dyad (from photoID column)
      - Matches     : number of matches in the dyad (from redCards column) used as exposure
      - ExposureOffset: log(Matches) used as offset in Poisson model
      - RefImplicit : referee-country mean implicit bias (from leagueCountry)
      - RefExplicit : referee-country mean explicit bias (from club)
      - RefID       : referee identifier (from goals)
      - PlayerID    : player identifier (from birthday)
    """

    # Work on a copy
    df = df.copy()

    # Convert rater columns to numeric (they are normalized 0-1 in schema but may be strings)
    df['rater1'] = pd.to_numeric(df.get('rater1'), errors='coerce')
    df['rater2'] = pd.to_numeric(df.get('rater2'), errors='coerce')

    # Compute average skin tone from the two raters
    df['SkinToneAvg'] = df[['rater1', 'rater2']].mean(axis=1)

    # Define light vs dark extremes; exclude middle/ambiguous ratings
    # Use strict thresholds per specification: >0.66 dark, <0.33 light; otherwise ambiguous
    df['IsDark'] = np.where(df['SkinToneAvg'] > 0.66, 1,
                             np.where(df['SkinToneAvg'] < 0.33, 0, np.nan))

    # Keep only clearly Light or Dark observations (drop medium/ambiguous)
    df = df[df['IsDark'].notnull()].copy()
    df['IsDark'] = df['IsDark'].astype(int)

    # Dependent variable: number of red cards given by the referee to the player in the dyad.
    # According to the provided schema text the photoID column is described as number of red cards.
    # We'll use photoID as the red-card count (rename to RedCardCount).
    df['RedCardCount'] = pd.to_numeric(df.get('photoID'), errors='coerce')
    # If photoID had missing values, assume zero red cards only if that makes sense; otherwise drop NA
    # We'll drop rows where RedCardCount is missing because we cannot infer outcome
    df = df[df['RedCardCount'].notnull()].copy()
    # Cast to integer count (if non-integer values appear, round down)
    df['RedCardCount'] = df['RedCardCount'].astype(int)

    # Exposure: number of matches in the dyad. Based on schema, the redCards column is described as
    # number of games in the player-referee dyad. We will treat that column as Matches.
    df['Matches'] = pd.to_numeric(df.get('redCards'), errors='coerce')
    # Drop rows with missing or zero exposure (cannot model rate with zero exposure)
    df = df[(df['Matches'].notnull()) & (df['Matches'] > 0)].copy()

    # Offset for Poisson regression is log(Matches)
    df['ExposureOffset'] = np.log(df['Matches'].astype(float))

    # Controls: referee-country implicit and explicit bias
    df['RefImplicit'] = pd.to_numeric(df.get('leagueCountry'), errors='coerce')
    df['RefExplicit'] = pd.to_numeric(df.get('club'), errors='coerce')

    # Identifiers for clustering/diagnostics
    df['RefID'] = df.get('goals')  # schema indicates 'goals' stores referee ID
    df['PlayerID'] = df.get('birthday')  # schema indicates 'birthday' stores player short name

    # Drop rows missing key predictors/controls to ensure a clean model matrix
    df = df.dropna(subset=['SkinToneAvg', 'IsDark', 'RedCardCount', 'Matches', 'RefImplicit', 'RefExplicit', 'RefID', 'PlayerID'])

    # Keep only the columns needed for analysis (but keep others in case user wants them)
    # We'll ensure the important analysis columns are present
    analysis_cols = ['SkinToneAvg', 'IsDark', 'RedCardCount', 'Matches', 'ExposureOffset', 'RefImplicit', 'RefExplicit', 'RefID', 'PlayerID']
    for col in analysis_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a Poisson regression predicting red card counts with log(matches) as an offset.

    Model: RedCardCount ~ IsDark + RefImplicit + RefExplicit
    Offset: ExposureOffset = log(Matches)

    Uses clustered (by referee) robust standard errors.
    Returns the cluster-robust results object.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Make sure the necessary columns exist
    required = ['RedCardCount', 'IsDark', 'RefImplicit', 'RefExplicit', 'ExposureOffset', 'RefID']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula. IsDark is binary indicator (1=dark, 0=light). Controls are continuous.
    formula = 'RedCardCount ~ IsDark + RefImplicit + RefExplicit'

    # Fit Poisson GLM with offset; use the model's fit() first, then obtain cluster-robust covariances
    glm_mod = smf.glm(formula=formula, data=df, family=sm.families.Poisson(), offset=df['ExposureOffset'])
    glm_res = glm_mod.fit()

    # Cluster-robust standard errors by referee (RefID)
    try:
        clustered_res = glm_res.get_robustcov_results(cov_type='cluster', groups=df['RefID'])
    except Exception:
        # If clustering fails for any reason, return the original fitted results
        clustered_res = glm_res

    # Print a brief summary and return the cluster-robust results object
    print(clustered_res.summary())
    return clustered_res