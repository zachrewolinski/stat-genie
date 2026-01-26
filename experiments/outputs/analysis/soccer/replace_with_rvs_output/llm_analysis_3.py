from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/replace_with_rvs_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for count regression.
    Produces the following final columns used by the model:
      - redCards: integer count (existing column)
      - games: integer exposure (existing column)
      - log_games: np.log(games) used as model offset
      - rater1, rater2: used to compute avg_rater
      - avg_rater: mean of rater1 and rater2
      - DarkSkin: binary (1 = dark, 0 = light); intermediate ratings removed
      - age: approximated age at 2013-01-01
      - height, weight, position, leagueCountry, yellowCards, yellowReds, goals, refNum, meanIAT, meanExp: kept for controls

    Notes:
    - We exclude dyads with missing rater scores or intermediate skin-tone ratings (to focus comparison dark vs light).
    - We exclude dyads with games <= 0 or missing redCards/games.
    """

    df = df.copy()

    # Keep relevant columns (will add/derive new ones)
    required_cols = [
        'redCards', 'games', 'rater1', 'rater2', 'birthday', 'height', 'weight',
        'position', 'leagueCountry', 'yellowCards', 'yellowReds', 'goals',
        'refNum', 'meanIAT', 'meanExp'
    ]
    # If some required columns are missing, proceed but operations will naturally raise errors for missing columns.

    # Drop rows with missing critical fields
    df = df.dropna(subset=['rater1', 'rater2', 'redCards', 'games'])

    # Keep only dyads with at least one game (exposure must be positive)
    df = df[df['games'] > 0]

    # Average rater score (normalized to [0,1] in the data description)
    df['avg_rater'] = df[['rater1', 'rater2']].mean(axis=1)

    # Dichotomize into Dark vs Light; exclude middle ratings to sharpen contrast
    # thresholds chosen to map approximately to bottom/middle/top thirds: <=0.33 light, >=0.66 dark
    df['DarkSkin'] = np.nan
    df.loc[df['avg_rater'] <= 0.33, 'DarkSkin'] = 0
    df.loc[df['avg_rater'] >= 0.66, 'DarkSkin'] = 1

    # Keep only clear dark or light cases
    df = df[df['DarkSkin'].isin([0, 1])].copy()

    # Parse birthday and compute approximate age at season midpoint (2013-01-01)
    # birthday format in schema: 'dd.mm.yyyy'
    df['birthday_dt'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y', errors='coerce')
    ref_date = pd.to_datetime('2013-01-01')
    df['age'] = (ref_date - df['birthday_dt']).dt.days / 365.25

    # Create log exposure for model offset; add a small constant guard (shouldn't be needed because games>0)
    df['log_games'] = np.log(df['games'].astype(float))

    # Keep numeric control columns as-is; coerce to numeric where appropriate
    for col in ['height', 'weight', 'yellowCards', 'yellowReds', 'goals', 'meanIAT', 'meanExp']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clean position and leagueCountry categories (fill missing with 'Unknown')
    if 'position' in df.columns:
        df['position'] = df['position'].fillna('Unknown').astype(str)
    if 'leagueCountry' in df.columns:
        df['leagueCountry'] = df['leagueCountry'].fillna('Unknown').astype(str)

    # Final selection of columns to return (model will create dummies as needed)
    final_cols = [
        'redCards', 'games', 'log_games', 'rater1', 'rater2', 'avg_rater', 'DarkSkin',
        'age', 'height', 'weight', 'yellowCards', 'yellowReds', 'goals',
        'position', 'leagueCountry', 'refNum', 'meanIAT', 'meanExp'
    ]

    # If any final column is missing from df (e.g., not present in the original), create it with NA to avoid KeyError
    for col in final_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a negative binomial GLM for redCards counts with log(games) as offset.
    Uses clustered robust standard errors at the referee level (refNum).

    Model formula conceptually: redCards ~ DarkSkin + age + height + weight + yellowCards + yellowReds + goals + C(position) + C(leagueCountry) + meanIAT + meanExp
    offset = log_games

    Returns:
      - results: the fitted GLMResults object from statsmodels
    """

    # Copy to avoid modifying caller's df
    df = df.copy()

    # Drop rows with missing outcome, exposure, or key predictors
    df = df.dropna(subset=['redCards', 'games', 'log_games', 'DarkSkin'])

    # Prepare design matrix: include controls and create dummies for categorical variables
    cat_cols = []
    if 'position' in df.columns:
        cat_cols.append('position')
    if 'leagueCountry' in df.columns:
        cat_cols.append('leagueCountry')

    # Base numeric covariates
    numeric_covs = [c for c in ['age', 'height', 'weight', 'yellowCards', 'yellowReds', 'goals', 'meanIAT', 'meanExp'] if c in df.columns]

    X = df[numeric_covs].copy() if numeric_covs else pd.DataFrame(index=df.index)

    # Add the main independent variable
    X['DarkSkin'] = df['DarkSkin'].astype(float)

    # Add categorical dummies (drop first to avoid multicollinearity)
    if cat_cols:
        dummies = pd.get_dummies(df[cat_cols].astype(str), drop_first=True)
        X = pd.concat([X, dummies], axis=1)

    # Fill remaining NA numeric covariates with their column means to allow estimation (alternatively could drop)
    for col in X.columns:
        if X[col].isnull().any():
            # If all values are missing, fill with zero
            if X[col].notnull().sum() == 0:
                X[col] = 0.0
            else:
                X[col] = X[col].fillna(X[col].mean())

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Response
    y = df['redCards'].astype(float)

    # Offset
    offset = df['log_games'].astype(float)

    # Fit Negative Binomial GLM
    # Use statsmodels.api.GLM with NegativeBinomial family
    try:
        fam = sm.families.NegativeBinomial()
        model_glm = sm.GLM(y, X, family=fam, offset=offset)
        # Fit with default method
        results = model_glm.fit()
        # Recompute robust clustered standard errors by referee (refNum) if available
        if 'refNum' in df.columns:
            clusters = df['refNum']
            # Use cov_type='cluster'
            results = model_glm.fit(cov_type='cluster', cov_kwds={'groups': clusters})
    except Exception:
        # Fallback to Poisson with robust SEs if NegativeBinomial fails
        fam = sm.families.Poisson()
        model_glm = sm.GLM(y, X, family=fam, offset=offset)
        results = model_glm.fit(cov_type='cluster', cov_kwds={'groups': df['refNum']}) if 'refNum' in df.columns else model_glm.fit()

    # Return the fitted results object (contains params, summary, etc.)
    return results


