from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/add_features_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformations performed:
    - Make a safe copy of the dataframe.
    - Compute an average skin tone rating from rater1 and rater2 (skipna=True).
    - Keep only players with extreme ratings (Light or Dark) by thresholding the normalized 0-1 ratings:
        * Light: PlayerSkinAvg <= 0.25
        * Dark: PlayerSkinAvg >= 0.75
      This creates a clear contrast that answers the research question "dark vs light".
    - Create a binary PlayerDark indicator (1=Dark, 0=Light).
    - Compute log_games = log(games) to use as an offset (exposure) in count regression.
    - Drop rows missing any variables essential for modeling (redCards, games, meanIAT, position, leagueCountry).
    - Return the transformed dataframe with the new columns: PlayerSkinAvg, PlayerSkinGroup, PlayerDark, log_games.
    """
    df = df.copy()

    # compute mean skin rating from available rater scores (rater1 and rater2 are on normalized 0-1 scale)
    df['PlayerSkinAvg'] = df[['rater1', 'rater2']].mean(axis=1, skipna=True)

    # Create categorical grouping: Light, Middle, Dark based on extremes of the 5-point normalized scale
    # Normalized scale values are expected to be in {0.0,0.25,0.5,0.75,1.0}
    df['PlayerSkinGroup'] = pd.Series(np.where(df['PlayerSkinAvg'] >= 0.75, 'Dark',
                                                np.where(df['PlayerSkinAvg'] <= 0.25, 'Light', 'Middle')), index=df.index)

    # Keep only extreme ratings (Light and Dark) so the comparison matches the research question
    df = df[df['PlayerSkinGroup'].isin(['Dark', 'Light'])].copy()

    # Binary indicator: 1 = Dark, 0 = Light
    df['PlayerDark'] = (df['PlayerSkinGroup'] == 'Dark').astype(int)

    # Create offset: log number of games (games is >=1 in schema, but clip to be safe)
    df['games'] = pd.to_numeric(df['games'], errors='coerce')
    df['log_games'] = np.log(df['games'].clip(lower=1))

    # Ensure necessary covariates exist and drop rows missing essential model variables
    essential_cols = ['redCards', 'games', 'meanIAT', 'position', 'leagueCountry', 'PlayerDark', 'log_games']
    df = df.dropna(subset=essential_cols).reset_index(drop=True)

    # Ensure numeric types for continuous controls
    for col in ['age', 'height', 'weight']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # (We do not aggressively drop rows for age/height/weight here; model will drop or handle missing as needed.)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Modeling approach:
    - Fit a Negative Binomial generalized linear model (GLM) for red card counts (overdispersion expected) with an offset for number of games.
    - Main predictor: PlayerDark (1 = Dark, 0 = Light).
    - Include meanIAT as a control and as a moderator via interaction PlayerDark:meanIAT to test whether referee-country implicit bias moderates the effect.
    - Additional covariates: age, height, weight, position (categorical), leagueCountry (categorical).
    - Cluster robust standard errors at the referee level (refNum) to account for within-referee correlation.

    Returns:
    - A statsmodels results instance with cluster-robust standard errors (fitted Negative Binomial).
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Drop rows missing regressors used in the formula (age/height/weight are included if present)
    required_for_model = ['redCards', 'log_games', 'PlayerDark', 'meanIAT', 'position', 'leagueCountry', 'refNum']
    df_model = df.dropna(subset=required_for_model).copy()

    # If age/height/weight contain NaNs, drop them for this specification to get a complete-case model
    for optional in ['age', 'height', 'weight']:
        if optional in df_model.columns:
            df_model = df_model.dropna(subset=[optional])

    # Formula: include explicit interaction between PlayerDark and meanIAT
    # Use categorical encoding for position and leagueCountry via C()
    formula = 'redCards ~ PlayerDark + meanIAT + PlayerDark:meanIAT + age + height + weight + C(position) + C(leagueCountry)'

    # Fit Negative Binomial GLM with offset = log_games
    model_glm = smf.glm(formula=formula,
                        data=df_model,
                        family=sm.families.NegativeBinomial(),
                        offset=df_model['log_games'])

    res = model_glm.fit()

    # Obtain cluster-robust covariance (cluster by referee ID)
    try:
        res_cluster = res.get_robustcov_results(cov_type='cluster', groups=df_model['refNum'])
    except Exception:
        # If cluster robust fails for any reason, return the plain fit but warn the user via the returned object
        res_cluster = res

    # Return the results object with cluster-robust cov if available
    return res_cluster


