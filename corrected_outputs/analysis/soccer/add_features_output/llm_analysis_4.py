from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/add_features_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure numeric columns are numeric
    df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce')
    df['games'] = pd.to_numeric(df['games'], errors='coerce')
    df['rater1'] = pd.to_numeric(df.get('rater1'), errors='coerce')
    df['rater2'] = pd.to_numeric(df.get('rater2'), errors='coerce')

    # Drop rows missing essential fields
    df = df.dropna(subset=['redCards', 'games'])

    # Construct average skin tone from raters (use available rater if one is missing)
    df['SkinToneAvg'] = df[['rater1', 'rater2']].mean(axis=1, skipna=True)
    df = df.dropna(subset=['SkinToneAvg'])

    # Create categorical skin tone grouping: Dark vs Light. Exclude middle-category (Medium) to get a clean comparison.
    # rater ratings are normalized to [0,1] where higher = darker
    def _skin_cat(x):
        if x >= 0.6:
            return 'Dark'
        elif x <= 0.4:
            return 'Light'
        else:
            return 'Medium'
    df['SkinToneCategory'] = df['SkinToneAvg'].apply(_skin_cat)

    # Keep only Dark and Light for the primary test
    df = df[df['SkinToneCategory'].isin(['Dark', 'Light'])]

    # Binary indicator for dark skin (1) vs light skin (0)
    df['DarkSkin'] = (df['SkinToneCategory'] == 'Dark').astype(int)

    # Exposure offset: log number of games in the dyad
    # ensure games > 0 (dataset minimum is 1, but defensively handle zeros)
    df = df[df['games'] > 0]
    df['offset'] = np.log(df['games'])

    # Convert several numeric predictors to numeric (if present)
    numeric_cols = ['age', 'height', 'weight', 'goals', 'yellowCards', 'meanIAT', 'meanExp']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Create standardized (z-scored) versions of player-level continuous controls used in the model
    for c in ['age', 'height', 'weight', 'goals', 'yellowCards']:
        if c in df.columns:
            denom = df[c].std(ddof=0)
            if denom == 0 or np.isnan(denom):
                df[c + '_z'] = (df[c] - df[c].mean()).fillna(0.0)
            else:
                df[c + '_z'] = (df[c] - df[c].mean()) / denom
        else:
            # If column not present, create NaN column so downstream code can check presence
            df[c + '_z'] = np.nan

    # Ensure categorical columns are of type category (helps get_dummies later)
    if 'position' in df.columns:
        df['position'] = df['position'].astype('category')
    if 'leagueCountry' in df.columns:
        df['leagueCountry'] = df['leagueCountry'].astype('category')

    # Keep a conservative set of columns required for modeling and diagnostics
    keep_cols = [
        'playerShort', 'refNum', 'redCards', 'games', 'offset',
        'DarkSkin', 'SkinToneAvg', 'SkinToneCategory',
        'position', 'leagueCountry', 'meanIAT', 'meanExp',
        'age_z', 'height_z', 'weight_z', 'goals_z', 'yellowCards_z'
    ]
    # Only return columns that exist in the input dataframe
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a Negative Binomial regression of redCards on DarkSkin (binary) with exposure offset = log(games).
    Controls: referee-country bias measures (meanIAT, meanExp), standardized player-level continuous controls,
    dummies for position and leagueCountry. Cluster-robust SEs by refNum.

    Returns: statsmodels results object (cluster-robust if refNum present)
    """
    df = df.copy()

    # Ensure the required outcome and offset exist
    if 'redCards' not in df.columns:
        raise ValueError('redCards column required in transformed dataframe')
    if 'offset' not in df.columns:
        raise ValueError('offset column (log(games)) required in transformed dataframe')

    # Base covariates list (only include those present in the dataframe)
    base_covs = ['DarkSkin', 'meanIAT', 'meanExp', 'age_z', 'height_z', 'weight_z', 'goals_z', 'yellowCards_z']
    base_covs = [c for c in base_covs if c in df.columns]

    # Create dummy variables for categorical controls (drop_first to avoid multicollinearity)
    cat_parts = []
    if 'position' in df.columns:
        pos_dummies = pd.get_dummies(df['position'], prefix='pos', drop_first=True)
        cat_parts.append(pos_dummies)
    if 'leagueCountry' in df.columns:
        country_dummies = pd.get_dummies(df['leagueCountry'], prefix='country', drop_first=True)
        cat_parts.append(country_dummies)

    # Construct design matrix X
    X_parts = [df[base_covs].fillna(0)]  # fill missing covariates with 0 (after standardization most missings become 0)
    X_parts += cat_parts
    if len(X_parts) > 1:
        X = pd.concat(X_parts, axis=1)
    else:
        X = X_parts[0]

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Outcome
    y = df['redCards']

    # Fit Negative Binomial GLM with offset
    model_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=df['offset'])
    res_nb = model_nb.fit()

    # If referee id is present, compute cluster-robust SEs clustered by refNum
    if 'refNum' in df.columns:
        try:
            res_nb_robust = res_nb.get_robustcov_results(cov_type='cluster', groups=df['refNum'])
            # Print summary for user's convenience (can be removed in programmatic use)
            print(res_nb_robust.summary())
            return res_nb_robust
        except Exception as e:
            # Fall back to non-robust results if clustering fails
            print('Cluster robust covariance computation failed, returning plain results. Error:', e)
            print(res_nb.summary())
            return res_nb
    else:
        print(res_nb.summary())
        return res_nb


