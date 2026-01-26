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
    Transform the raw dyad dataset into a dataframe ready for count regression (negative binomial)
    Model-ready columns produced:
      - skin_score: continuous mean of rater1 and rater2 (0-1, higher = darker)
      - DarkSkin: median-split binary indicator (1 = dark)
      - log_games: log(games) used as offset/exposure
      - skin_score_x_meanIAT: interaction term for moderation tests
    The function drops rows missing the minimum required data (games, redCards, rater1, rater2).
    It imputes medians for numeric controls if they are missing (age, height, weight, meanIAT, meanExp).
    """
    df = df.copy()

    # Drop rows missing critical outcome/exposure/IV data
    df = df.dropna(subset=['games', 'redCards', 'rater1', 'rater2'])

    # Keep only dyads with at least one game (games is exposure)
    df = df[df['games'] > 0].copy()

    # Continuous skin score: mean of the two independent raters
    df['skin_score'] = df[['rater1', 'rater2']].mean(axis=1)

    # Binary dark/light skin indicator (median split). This is for robustness checks.
    median_skin = df['skin_score'].median()
    df['DarkSkin'] = (df['skin_score'] >= median_skin).astype(int)

    # Exposure offset: log number of games (use natural log)
    # games should be > 0 because we filtered earlier
    df['log_games'] = np.log(df['games'].astype(float))

    # Ensure numeric controls exist and impute medians when missing
    for col in ['age', 'height', 'weight', 'meanIAT', 'meanExp']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median()
            # If entire column is NaN, fill with 0 to avoid downstream errors
            if np.isnan(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

    # Interaction term for moderation tests (skin x implicit-country-bias)
    if 'meanIAT' in df.columns:
        df['skin_score_x_meanIAT'] = df['skin_score'] * df['meanIAT']
    else:
        # create a column of zeros if meanIAT not available so model code can run without branching
        df['skin_score_x_meanIAT'] = 0.0

    # Keep a compact set of columns necessary for modeling and diagnostics
    wanted = [
        'playerShort', 'player', 'refNum', 'refCountry',
        'games', 'redCards', 'rater1', 'rater2',
        'skin_score', 'DarkSkin', 'log_games', 'skin_score_x_meanIAT',
        'age', 'height', 'weight', 'meanIAT', 'meanExp',
        'position', 'leagueCountry'
    ]

    keep_cols = [c for c in wanted if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a Negative Binomial GLM for red card counts with games as exposure.

    Model specification (primary):
      redCards ~ skin_score + age + height + weight + meanIAT + meanExp + skin_score_x_meanIAT + position dummies + leagueCountry dummies
    Offset: log_games (so the model estimates rates per game)
    Standard errors clustered at the referee level (refNum).

    Returns:
      A dictionary with keys:
        - 'model': fitted statsmodels result object
        - 'rate_ratios': DataFrame with exponentiated coefficients (incidence rate ratios) and 95% CI
    """
    df = df.copy()

    # Numerical predictors used directly
    numeric_predictors = [c for c in ['skin_score', 'age', 'height', 'weight', 'meanIAT', 'meanExp', 'skin_score_x_meanIAT'] if c in df.columns]
    exog = df[numeric_predictors].copy()

    # Convert categorical controls to dummies
    # position (many categories) -> dummies, drop first to avoid multicollinearity
    if 'position' in df.columns:
        pos_dums = pd.get_dummies(df['position'].astype(str), prefix='position', drop_first=True)
        exog = pd.concat([exog, pos_dums], axis=1)

    # leagueCountry -> dummies
    if 'leagueCountry' in df.columns:
        league_dums = pd.get_dummies(df['leagueCountry'].astype(str), prefix='league', drop_first=True)
        exog = pd.concat([exog, league_dums], axis=1)

    # Add constant
    exog = sm.add_constant(exog, has_constant='add')

    # Fit Negative Binomial GLM with offset = log_games
    # Use clustered standard errors by refNum (referee) to account for non-independence across dyads with the same referee
    offset = df['log_games'] if 'log_games' in df.columns else np.log(df['games'].clip(lower=1))

    nb_model = sm.GLM(df['redCards'], exog, family=sm.families.NegativeBinomial(), offset=offset)
    try:
        res = nb_model.fit(cov_type='cluster', cov_kwds={'groups': df['refNum']})
    except Exception:
        # Fallback to default (non-clustered) if clustering fails for any reason
        res = nb_model.fit()

    # Exponentiate coefficients to get incidence rate ratios (IRR)
    rr = np.exp(res.params)
    conf = res.conf_int()
    conf_exp = np.exp(conf)

    rr_df = pd.DataFrame({
        'irr': rr,
        'ci_lower': conf_exp[0],
        'ci_upper': conf_exp[1]
    })

    results = {
        'model': res,
        'rate_ratios': rr_df
    }

    return results


