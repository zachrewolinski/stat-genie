from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/noperturb_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a working copy
    df = df.copy()

    # Required raw columns: rater1, rater2, redCards, games, birthday, goals, yellowCards, yellowReds,
    # position, leagueCountry, meanIAT, meanExp, refNum

    # 1) Drop rows with missing critical values
    df = df.dropna(subset=['rater1', 'rater2', 'redCards', 'games', 'birthday'])

    # 2) Ensure games > 0 (cannot have exposure 0); drop if games <= 0
    df = df[df['games'] > 0]

    # 3) Compute averaged skin tone from the two independent raters
    df['SkinToneAvg'] = df[['rater1', 'rater2']].mean(axis=1)

    # The rater scale was normalized to 1 across 5 categories giving values like 0,0.25,0.5,0.75,1
    # Define extremes to compare dark vs light: keep only players at the 'light' and 'dark' extremes
    df['SkinCategory'] = 'Medium'
    df.loc[df['SkinToneAvg'] <= 0.25, 'SkinCategory'] = 'Light'
    df.loc[df['SkinToneAvg'] >= 0.75, 'SkinCategory'] = 'Dark'

    # Filter to light and dark extremes only (exclude medium/ambiguous ratings)
    df = df[df['SkinCategory'].isin(['Light', 'Dark'])]

    # Binary indicator: DarkSkin = 1 if Dark, 0 if Light
    df['DarkSkin'] = (df['SkinCategory'] == 'Dark').astype(int)

    # 4) Create per-game control variables to reflect rates in the dyad
    # Prevent division by zero is already handled by games>0 filter
    df['goals_per_game'] = df['goals'] / df['games']
    df['yellowCards_per_game'] = df['yellowCards'] / df['games']
    df['yellowReds_per_game'] = df['yellowReds'] / df['games']

    # 5) Parse birthday and compute approximate age at season midpoint (use Jan 1, 2013 as reference)
    # Birthday format in schema is dd.mm.yyyy
    df['birthday'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y', errors='coerce')
    season_ref = pd.to_datetime('2013-01-01')
    df['age_years'] = (season_ref - df['birthday']).dt.days / 365.25

    # 6) Keep required columns for modeling and drop any rows with now-missing values in these
    required_final = [
        'playerShort', 'redCards', 'games', 'DarkSkin', 'goals_per_game',
        'yellowCards_per_game', 'yellowReds_per_game', 'age_years',
        'position', 'leagueCountry', 'meanIAT', 'meanExp', 'refNum'
    ]
    df = df.dropna(subset=required_final)

    # 7) Ensure numeric types for critical fields
    numeric_cols = ['redCards', 'games', 'DarkSkin', 'goals_per_game',
                    'yellowCards_per_game', 'yellowReds_per_game', 'age_years',
                    'meanIAT', 'meanExp', 'refNum']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Final drop if conversion introduced NaNs
    df = df.dropna(subset=numeric_cols)

    # Reset index and return
    df = df.reset_index(drop=True)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    import numpy as np

    # The transform function should already have prepared the following columns exactly:
    # 'redCards' (count DV), 'games' (exposure), 'DarkSkin' (0/1 IV),
    # 'goals_per_game', 'yellowCards_per_game', 'yellowReds_per_game', 'age_years',
    # 'position' (categorical), 'leagueCountry' (categorical), 'meanIAT', 'meanExp', 'refNum'

    # Build formula: Negative binomial GLM with offset = log(games)
    formula = (
        'redCards ~ DarkSkin + goals_per_game + yellowCards_per_game + '
        'yellowReds_per_game + age_years + meanIAT + meanExp + C(position) + C(leagueCountry)'
    )

    # Fit GLM Negative Binomial with offset for games (exposure). Using the log of games as offset.
    model_glm = smf.glm(formula=formula,
                        data=df,
                        family=sm.families.NegativeBinomial(),
                        offset=np.log(df['games']))

    res = model_glm.fit()

    # Obtain cluster-robust standard errors clustered by referee (refNum) to account for referee-level correlation
    # (population-averaged inference about the association between skin tone and red cards)
    try:
        res_cluster = res.get_robustcov_results(cov_type='cluster', groups=df['refNum'])
    except Exception:
        # Fallback: if clustering fails, return the original fit
        res_cluster = res

    # Print summary of clustered results for user inspection
    print(res_cluster.summary())

    # Return the fitted result with clustered covariances (or original fit if clustering not available)
    return res_cluster


