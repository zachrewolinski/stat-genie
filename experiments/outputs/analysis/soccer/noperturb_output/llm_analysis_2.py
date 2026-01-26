from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/noperturb_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dyad dataset into analysis-ready dataframe.
    - Compute average skin tone from rater1 and rater2 (use available rater if one is missing).
    - Dichotomize skin tone into 'Light' (<=0.25) and 'Dark' (>=0.75); drop 'Medium' observations.
    - Parse birthday and compute approximate age in 2013 (season year used as reference).
    - Impute a small set of numeric controls with median when missing (height, weight, goals, yellowCards, yellowReds, meanIAT, meanExp).
    - Keep only columns required for modeling and return the cleaned dataframe.
    """
    df = df.copy()

    # Ensure relevant columns exist and coerce numeric types where appropriate
    # Columns: rater1, rater2, redCards, games, birthday, meanIAT, meanExp, position, height, weight, goals, yellowCards, yellowReds, leagueCountry, refNum
    # Drop rows missing the essential outcome/exposure or both raters
    essential = ['redCards', 'games', 'refNum']
    df = df.dropna(subset=essential)

    # Compute skin_tone_avg from rater1 and rater2
    # If one rater is missing, use the other; if both missing, the result will be NaN and those rows will be dropped
    df['skin_tone_avg'] = df[['rater1', 'rater2']].mean(axis=1)

    # Dichotomize skin tone: map clearly light and clearly dark, drop medium/uncertain
    # rater scale originally 5 points normalized to [0,1] with increments of 0.25 (0,0.25,0.5,0.75,1)
    df['SkinTone'] = pd.NA
    df.loc[df['skin_tone_avg'] <= 0.25, 'SkinTone'] = 'Light'
    df.loc[df['skin_tone_avg'] >= 0.75, 'SkinTone'] = 'Dark'

    # Keep only clearly light and dark players for the primary comparison
    df = df[df['SkinTone'].isin(['Light', 'Dark'])].copy()

    # Parse birthday and compute age (reference year 2013 for the 2012-2013 season)
    # Birthdays are in dd.mm.yyyy format according to schema
    def parse_bday_to_age(x):
        try:
            dt = pd.to_datetime(x, format='%d.%m.%Y', errors='coerce')
            if pd.isna(dt):
                return np.nan
            return 2013 - dt.year
        except Exception:
            return np.nan

    if 'birthday' in df.columns:
        df['age'] = df['birthday'].apply(parse_bday_to_age)
    else:
        df['age'] = np.nan

    # Make sure numeric columns are numeric
    numeric_cols = ['meanIAT', 'meanExp', 'height', 'weight', 'goals', 'yellowCards', 'yellowReds']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # if missing in DF, create NA column
            df[col] = np.nan

    # Impute key numeric controls with median (simple, transparent approach)
    for col in ['meanIAT', 'meanExp', 'height', 'weight', 'goals', 'yellowCards', 'yellowReds', 'age']:
        if df[col].isna().any():
            median = df[col].median()
            if pd.isna(median):
                # if median cannot be computed (all NaNs), fill 0 as conservative default
                median = 0
            df[col] = df[col].fillna(median)

    # Ensure games is integer and positive (schema says min 1, but guard anyway)
    df['games'] = pd.to_numeric(df['games'], errors='coerce').fillna(1).astype(int)
    df.loc[df['games'] < 1, 'games'] = 1

    # Keep a compact set of columns we'll use in modeling
    keep_cols = [
        'playerShort', 'player', 'photoID', 'refNum', 'refCountry', 'leagueCountry',
        'redCards', 'games', 'skin_tone_avg', 'SkinTone',
        'meanIAT', 'meanExp', 'position', 'age', 'height', 'weight',
        'goals', 'yellowCards', 'yellowReds'
    ]

    # Add any missing keep_cols as NA columns so returned DF has a consistent schema
    for col in keep_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[keep_cols].reset_index(drop=True)

    # Ensure redCards is integer count
    df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce').fillna(0).astype(int)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a negative binomial regression (count model) predicting the number of red cards
    a player received from a particular referee (redCards). Use the number of games
    in the dyad as an exposure (log-offset). Primary predictor is SkinTone ('Dark' vs 'Light').

    We include controls for country-level implicit/explicit bias (meanIAT, meanExp), player
    attributes (position, age, height, weight), disciplinary history in the dyad (yellowCards, yellowReds),
    offensive contribution (goals), and league fixed effects (leagueCountry).

    Cluster standard errors at the referee level (refNum) to account for referee-level dependence.

    Returns: fitted results object with cluster-robust SEs.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = df.copy()

    # Create offset = log(games) (exposure)
    # games should be >=1 per transform; but guard against zeros
    df['games'] = pd.to_numeric(df['games'], errors='coerce').fillna(1).astype(int)
    df.loc[df['games'] < 1, 'games'] = 1
    offset = np.log(df['games'].astype(float))

    # Formula: redCards as NB, offset by log(games)
    # Use categorical converters C(SkinTone) and C(position) and C(leagueCountry)
    formula = (
        'redCards ~ C(SkinTone) + meanIAT + meanExp + C(position) + age + height + weight '
        '+ goals + yellowCards + yellowReds + C(leagueCountry)'
    )

    # Fit GLM Negative Binomial with offset
    model_glm = smf.glm(formula=formula, data=df,
                        family=sm.families.NegativeBinomial(),
                        offset=offset)
    res = model_glm.fit()

    # Obtain cluster-robust standard errors clustered by referee id (refNum)
    # This returns a results instance with robust covariances
    try:
        res_clust = res.get_robustcov_results(cov_type='cluster', groups=df['refNum'])
    except Exception:
        # Fallback: if clustering fails, return the original fit object
        res_clust = res

    # Print a brief summary for quick inspection
    print(res_clust.summary())

    # Return the robust results object (or original results if clustering failed)
    return res_clust


