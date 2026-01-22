from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/positive_leading_statement_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dyad-level dataframe to variables used in the count regression.

    Steps performed:
    - Drop rows missing essential variables (rater1, rater2, redCards, games, refNum).
    - Create SkinScore as the mean of rater1 and rater2 (both raters provided on a 5-point scale then normalized to 0-1 in the raw data; the mean therefore ranges 0-1, higher = darker).
    - Bin SkinScore into three groups (Light, Mid, Dark) using cut points that isolate clearly light and clearly dark ratings. Keep only Light and Dark groups to obtain a clear binary comparison.
    - Create binary DarkSkin indicator (1 = Dark, 0 = Light).
    - Parse birthday and compute age at 2012-09-01 (season reference date). Impute median age if birthday parsing fails.
    - Copy height and weight to explicit column names height_cm and weight_kg.
    - Create lnGames which is the natural log of games and will be used as an offset in Poisson/Negative Binomial regressions.
    - Return a dataframe with only the columns required for modeling.
    """
    df = df.copy()

    # drop rows missing essential fields
    df = df.dropna(subset=['rater1', 'rater2', 'redCards', 'games', 'refNum'])

    # compute average skin score (rater values already normalized to a 0..1 scale in the raw data)
    df['SkinScore'] = df[['rater1', 'rater2']].mean(axis=1)

    # categorize into Light / Mid / Dark. Cut thresholds chosen to keep clear Light and Dark groups and exclude middle/ambiguous cases.
    df['SkinGroup'] = pd.cut(df['SkinScore'], bins=[-0.01, 0.40, 0.60, 1.01], labels=['Light', 'Mid', 'Dark'])

    # Keep only clearly Light or Dark players to make a focused comparison
    df = df[df['SkinGroup'].isin(['Light', 'Dark'])].copy()

    # binary indicator: Dark = 1, Light = 0
    df['DarkSkin'] = (df['SkinGroup'] == 'Dark').astype(int)

    # parse birthdays and compute age at season reference date
    df['birthday'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y', errors='coerce')
    season_ref = pd.to_datetime('2012-09-01')
    df['age_years'] = (season_ref - df['birthday']).dt.days / 365.25
    df['age_years'] = df['age_years'].fillna(df['age_years'].median())

    # explicit column names for anthropometrics
    df['height_cm'] = df['height']
    df['weight_kg'] = df['weight']

    # exposure for count models: log of games (games >= 1 in schema, but guard anyway)
    df['games'] = df['games'].fillna(1)
    df['games'] = df['games'].replace(0, 1)
    df['lnGames'] = np.log(df['games'])

    # tidy categorical controls
    df['position'] = df['position'].fillna('Unknown')
    df['leagueCountry'] = df['leagueCountry'].fillna('Unknown')

    # keep only the columns needed for modeling
    cols = [
        'playerShort', 'refNum', 'refCountry',
        'redCards', 'games', 'lnGames',
        'SkinScore', 'SkinGroup', 'DarkSkin',
        'age_years', 'height_cm', 'weight_kg',
        'meanIAT', 'meanExp',
        'position', 'leagueCountry'
    ]

    # ensure all columns exist (if meanIAT/meanExp missing in some rows, keep them as NA for model - statsmodels will handle or we could impute)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit count regression models to test whether darker-skinned players receive more red cards than lighter-skinned players.

    Modeling strategy:
    - Primary model: Negative Binomial regression (accounts for overdispersion) with an offset equal to lnGames (exposure).
    - Secondary / sensitivity model: Poisson regression with the same offset.
    - Include controls: age_years, height_cm, weight_kg, meanIAT, meanExp, categorical position and leagueCountry.
    - Use clustered robust standard errors at the referee level (refNum) to account for non-independence of observations judged by the same referee.

    The function returns a dictionary containing both fitted results objects with clustered robust covariances applied.
    """
    import statsmodels.formula.api as smf
    # formula: main effect is DarkSkin (1 = dark, 0 = light)
    formula = (
        'redCards ~ DarkSkin + age_years + height_cm + weight_kg + meanIAT + meanExp '
        '+ C(position) + C(leagueCountry)'
    )

    # Drop rows with missing values in model variables (statsmodels cannot handle NA in predictors). We keep rows with NA in meanIAT/meanExp only if present;
    model_df = df.dropna(subset=['redCards', 'lnGames', 'DarkSkin', 'age_years', 'height_cm', 'weight_kg', 'position', 'leagueCountry', 'refNum'])

    # Negative Binomial with offset = lnGames
    nb_model = smf.glm(formula=formula, data=model_df, family=sm.families.NegativeBinomial(), offset=model_df['lnGames']).fit()
    # Clustered robust SEs by refNum
    try:
        nb_clustered = nb_model.get_robustcov_results(cov_type='cluster', groups=model_df['refNum'])
    except Exception:
        # fallback to default (non-clustered) if clustering fails
        nb_clustered = nb_model

    # Poisson sensitivity model
    pois_model = smf.glm(formula=formula, data=model_df, family=sm.families.Poisson(), offset=model_df['lnGames']).fit()
    try:
        pois_clustered = pois_model.get_robustcov_results(cov_type='cluster', groups=model_df['refNum'])
    except Exception:
        pois_clustered = pois_model

    # Prepare a simple summary dictionary focusing on the DarkSkin coefficient
    summary_dict = {
        'n_obs': int(model_df.shape[0]),
        'n_referees': int(model_df['refNum'].nunique()),
        'neg_binom_result': nb_clustered,
        'poisson_result': pois_clustered
    }

    return summary_dict


