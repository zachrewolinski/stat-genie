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
    Transform raw dyad-level dataframe into analytic dataframe for testing whether dark-skinned players
    receive more red cards than light-skinned players.

    Outputs (columns required by later model):
      - redCards (int): count outcome
      - games (int): exposure (used as offset)
      - rater1, rater2 (float): original raters (kept for traceability)
      - skin_tone_avg (float): average of rater1 and rater2 on 0-1 normalized scale
      - SkinToneCat (category): 'Light' / 'Mid' / 'Dark'
      - IsDark (int): 1 if Dark, 0 if Light (we filter to only Light and Dark)
      - position, height, weight, goals, yellowCards, meanIAT, meanExp, leagueCountry, refNum
    """
    df = df.copy()

    # Ensure required columns exist
    required_cols = ['rater1', 'rater2', 'redCards', 'games', 'position', 'height', 'weight',
                     'goals', 'yellowCards', 'meanIAT', 'meanExp', 'leagueCountry', 'refNum']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe missing required columns: {missing}")

    # Drop rows with missing core values
    df = df.dropna(subset=['rater1', 'rater2', 'redCards', 'games', 'position', 'refNum'])

    # Make sure numeric types are correct
    df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce').fillna(0).astype(int)
    df['games'] = pd.to_numeric(df['games'], errors='coerce')

    # Remove dyads with non-positive games (games is used as exposure/offset)
    df = df[df['games'] > 0].copy()

    # Compute average skin tone rating (rater scale is normalized to 1 in source)
    df['skin_tone_avg'] = (df['rater1'].astype(float) + df['rater2'].astype(float)) / 2.0

    # Create categorical skin tone groups for a focused comparison between 'Light' and 'Dark'
    # rater normalized to 0..1 corresponding to 5-point original scale; thresholds chosen to pick
    # the bottom and top categories (>=0.75 approximates 'very dark', <=0.25 approximates 'very light').
    def tone_cat(x):
        if pd.isna(x):
            return pd.NA
        if x >= 0.75:
            return 'Dark'
        elif x <= 0.25:
            return 'Light'
        else:
            return 'Mid'

    df['SkinToneCat'] = df['skin_tone_avg'].apply(tone_cat).astype('category')

    # Binary indicator for main comparison (1 = Dark, 0 = Light). We'll restrict sample to rows that are either Dark or Light.
    df = df[df['SkinToneCat'].isin(['Dark', 'Light'])].copy()
    df['IsDark'] = (df['SkinToneCat'] == 'Dark').astype(int)

    # Additional derived variable: any_red (binary) for sensitivity analyses
    df['any_red'] = (df['redCards'] > 0).astype(int)

    # Ensure categorical variables are typed as category
    df['position'] = df['position'].astype('category')
    df['leagueCountry'] = df['leagueCountry'].astype('category')

    # Keep only columns necessary for the modeling to avoid accidental reliance on other columns
    keep_cols = [
        'redCards', 'any_red', 'games', 'IsDark', 'skin_tone_avg', 'SkinToneCat',
        'position', 'height', 'weight', 'goals', 'yellowCards', 'meanIAT', 'meanExp',
        'leagueCountry', 'refNum'
    ]

    # Some of these columns may not exist in all data exports (e.g., goals or yellowCards). If missing, create NA columns.
    for c in keep_cols:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit primary Negative Binomial model of redCards with games as exposure (offset).
    Provide clustered (by refNum) robust standard errors. Also run sensitivity logistic
    regression on any_red (binary) with clustered SEs.

    Returns a dictionary containing:
      - nb_model: fitted GLM negative binomial results (MLE)
      - nb_model_cluster: negative binomial results with cluster-robust SEs (referee clusters)
      - logit_model: fitted logistic regression (MLE) on any_red
      - logit_model_cluster: logistic regression results with cluster-robust SEs
    """
    import numpy as np
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Basic formula: effect of IsDark controlling for position, height, weight, goals, yellowCards,
    # and country-level bias measures. League country included as categorical fixed effect.
    formula_base = (
        'redCards ~ IsDark + C(position) + height + weight + goals + yellowCards + '
        'meanIAT + meanExp + C(leagueCountry)'
    )

    # Fit negative binomial GLM with log(games) as offset (exposure)
    # Note: statsmodels expects the offset argument to be an array (log of exposure).
    offset = np.log(df['games'].astype(float))

    nb_model = smf.glm(formula=formula_base, data=df,
                       family=sm.families.NegativeBinomial(),
                       offset=offset).fit()

    # Obtain cluster-robust covariance (clustered by refNum)
    try:
        nb_cluster = nb_model.get_robustcov_results(cov_type='cluster', groups=df['refNum'])
    except Exception:
        # Fallback: return the original model if clustering fails
        nb_cluster = nb_model

    # Sensitivity analysis: logistic regression on whether the player received ANY red card
    logit_formula = (
        'any_red ~ IsDark + C(position) + height + weight + goals + yellowCards + '
        'meanIAT + meanExp + C(leagueCountry)'
    )
    # Drop rows with missing any_red
    df_logit = df.dropna(subset=['any_red'])
    logit_model = smf.logit(formula=logit_formula, data=df_logit).fit(disp=False)
    try:
        logit_cluster = logit_model.get_robustcov_results(cov_type='cluster', groups=df_logit['refNum'])
    except Exception:
        logit_cluster = logit_model

    # Return objects (caller can inspect .summary() or coef tables).
    results = {
        'nb_model': nb_model,
        'nb_model_cluster': nb_cluster,
        'logit_model': logit_model,
        'logit_model_cluster': logit_cluster
    }

    return results


