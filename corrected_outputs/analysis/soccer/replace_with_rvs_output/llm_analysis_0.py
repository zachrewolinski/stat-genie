from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/replace_with_rvs_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to create variables required for modeling.

    Outputs (columns added or ensured):
      - skin_avg: average of rater1 and rater2 (continuous 0-1)
      - dark_vs_light: binary 1 for dark, 0 for light (rows with ambiguous/mid values removed)
      - age: player age in years measured at 2013-01-01 (season reference)
      - log_games: natural log of games (used as offset). Rows with games <= 0 are dropped.

    Rows with missing critical fields (rater1/rater2, redCards, games) are dropped.
    """
    # Make a copy to avoid modifying input in place
    df = df.copy()

    # Ensure numeric columns are numeric
    for col in ['rater1', 'rater2', 'redCards', 'games', 'height', 'weight', 'meanIAT', 'meanExp']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute average skin rating across two independent raters
    if 'rater1' in df.columns and 'rater2' in df.columns:
        df['skin_avg'] = df[['rater1', 'rater2']].mean(axis=1)
    else:
        df['skin_avg'] = np.nan

    # Drop rows missing the key variables for the primary analysis
    df = df.dropna(subset=['skin_avg', 'redCards', 'games'])

    # Remove dyads with zero or negative games (no exposure)
    df = df[df['games'] > 0]

    # Create a binary dark_vs_light variable:
    # - light if skin_avg <= 0.40
    # - dark if skin_avg >= 0.60
    # - drop middle values (0.40 < skin_avg < 0.60) to focus contrast
    df['dark_vs_light'] = np.nan
    df.loc[df['skin_avg'] <= 0.40, 'dark_vs_light'] = 0
    df.loc[df['skin_avg'] >= 0.60, 'dark_vs_light'] = 1
    df = df.dropna(subset=['dark_vs_light'])
    # Ensure integer type
    df['dark_vs_light'] = df['dark_vs_light'].astype(int)

    # Convert birthday to datetime and compute age at reference date (season midpoint)
    # Birthday format in schema is 'dd.mm.yyyy'. Use dayfirst=True.
    if 'birthday' in df.columns:
        df['birthday'] = pd.to_datetime(df['birthday'], dayfirst=True, errors='coerce')
    else:
        df['birthday'] = pd.NaT

    # Reference date: 2013-01-01 (approx. midpoint of 2012-2013 season)
    ref_date = pd.to_datetime('2013-01-01')
    df['age'] = (ref_date - df['birthday']).dt.days / 365.25

    # Compute log of games for use as offset in count models
    # Add a small constant is not necessary because games>0 after filtering
    df['log_games'] = np.log(df['games'])

    # Keep columns necessary for analysis and modeling; others are preserved but not required
    required_cols = [
        'playerShort', 'player', 'club', 'leagueCountry', 'birthday', 'age',
        'height', 'weight', 'position', 'games', 'log_games', 'redCards',
        'photoID', 'rater1', 'rater2', 'skin_avg', 'dark_vs_light',
        'refNum', 'refCountry', 'meanIAT', 'nIAT', 'seIAT', 'meanExp', 'nExp', 'seExp'
    ]
    # Some may not exist in input; keep those that do
    available = [c for c in required_cols if c in df.columns]
    # Return dataframe with available columns (keeps other columns too, but ensures these exist)
    return df[available + [c for c in df.columns if c not in available]]


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a negative binomial generalized linear model predicting redCards from skin tone
    (dark_vs_light) with an offset for number of games (exposure) and a set of controls.

    Returns:
      - results_robust: model results with cluster-robust standard errors clustered by refNum.

    Model specification (formula):
      redCards ~ dark_vs_light + skin_avg + meanIAT + meanExp + height + weight + age + C(position) + C(leagueCountry)
    Family: Negative Binomial
    Offset: log_games
    Robust SEs clustered by refNum
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Keep only rows required by the model and drop any remaining NA in model vars
    model_vars = [
        'redCards', 'dark_vs_light', 'skin_avg', 'meanIAT', 'meanExp',
        'height', 'weight', 'age', 'position', 'leagueCountry', 'log_games', 'refNum'
    ]
    # Ensure columns exist
    available = [v for v in model_vars if v in df.columns]
    df_model = df[available].dropna()

    # Ensure refNum is treated as grouping variable for clustering
    df_model['refNum'] = df_model['refNum'].astype(int)

    # Formula: include categorical controls using C()
    formula = (
        'redCards ~ dark_vs_light + skin_avg + meanIAT + meanExp + '
        'height + weight + age + C(position) + C(leagueCountry)'
    )

    # Fit Negative Binomial GLM with offset = log_games (exposure)
    # Using statsmodels' formula API
    glm_nb = smf.glm(formula=formula,
                    data=df_model,
                    family=sm.families.NegativeBinomial(),
                    offset=df_model['log_games'])

    res = glm_nb.fit()

    # Get cluster-robust standard errors clustered by referee (refNum)
    # If clustering fails for some reason (e.g., single cluster), fall back to default
    try:
        results_robust = res.get_robustcov_results(cov_type='cluster', groups=df_model['refNum'])
    except Exception:
        # return the original fit if robust clustering not possible
        results_robust = res

    # Print a short summary for quick inspection (caller can further inspect returned object)
    try:
        print(results_robust.summary())
    except Exception:
        pass

    return results_robust


