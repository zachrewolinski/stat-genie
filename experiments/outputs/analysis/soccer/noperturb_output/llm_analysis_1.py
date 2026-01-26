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
    Transform the raw dataset into a dataframe ready for analysis.

    Steps:
    - Drop observations missing critical variables (rater scores, redCards, games, refNum).
    - Exclude dyads with zero games (cannot compute an exposure offset).
    - Compute mean skin rating from two raters ('skin_mean').
    - Create three-level SkinGroup ('Dark', 'Light', 'Neither') using conservative cutoffs to isolate clear dark vs light ratings.
        * Raters are normalized to 1 (5-point scale mapped to 0, 0.25, 0.5, 0.75, 1.0). We define 'Dark' as mean >= 0.625, 'Light' as mean <= 0.375; middle values are 'Neither'.
    - Keep only 'Dark' and 'Light' players to obtain a focused comparison between dark and light-skinned players.
    - Create binary indicator SkinDark (1 = Dark, 0 = Light).
    - Parse birthday and compute age at 2013-01-01 (season mid/afterseason reference).
    - Drop remaining rows with missing key control variables (meanIAT, meanExp, position, leagueCountry, age).
    - Return a dataframe containing the modeling columns plus diagnostics (skin_mean, SkinGroup).
    """

    df = df.copy()

    # Drop rows missing rater scores, redCards, games, or referee id
    df = df.dropna(subset=['rater1', 'rater2', 'redCards', 'games', 'refNum'])

    # Remove dyads with zero games (cannot use log(games) as offset)
    df = df[df['games'] > 0].copy()

    # Compute mean skin rating from two independent raters
    df['skin_mean'] = (df['rater1'].astype(float) + df['rater2'].astype(float)) / 2.0

    # Classify skin groups: conservative cutoffs to isolate clear Dark vs Light
    # Ratings are normalized to 1 (possible values ~ 0.0,0.25,0.5,0.75,1.0)
    df['SkinGroup'] = np.where(
        df['skin_mean'] >= 0.625, 'Dark',
        np.where(df['skin_mean'] <= 0.375, 'Light', 'Neither')
    )

    # Keep only clearly Dark and Light players (drop 'Neither') so analysis is a direct contrast
    df = df[df['SkinGroup'].isin(['Dark', 'Light'])].copy()

    # Binary indicator for dark skin
    df['SkinDark'] = (df['SkinGroup'] == 'Dark').astype(int)

    # Parse birthday (format dd.mm.yyyy) to compute age at 2013-01-01
    df['birthday'] = pd.to_datetime(df['birthday'], dayfirst=True, errors='coerce')
    reference_date = pd.to_datetime('2013-01-01')
    df['age'] = (reference_date - df['birthday']).dt.days / 365.25

    # Drop rows missing key country-level bias measures or essential covariates
    df = df.dropna(subset=['meanIAT', 'meanExp', 'position', 'leagueCountry', 'age'])

    # Ensure numeric types for controls
    df['height'] = pd.to_numeric(df['height'], errors='coerce')
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df = df.dropna(subset=['height', 'weight'])

    # Keep only the columns required for modeling (plus skin diagnostics)
    out_cols = [
        'redCards', 'games', 'SkinDark', 'meanIAT', 'meanExp',
        'age', 'height', 'weight', 'position', 'leagueCountry', 'refNum',
        'skin_mean', 'SkinGroup'
    ]

    return df[out_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a negative binomial regression for the rate of red cards per game, comparing dark- vs light-skinned players.

    Model specification:
    - Outcome: redCards (count)
    - Exposure: games (offset = log(games))
    - Key predictor: SkinDark (1 = Dark, 0 = Light)
    - Controls: meanIAT, meanExp, age, height, weight, position (categorical), leagueCountry (categorical)
    - Clustered (referee-level) robust standard errors using refNum to account for non-independence across dyads involving the same referee.

    Returns a dictionary with the fitted model (with clustered SEs) and a small table of incidence rate ratios (IRRs) with 95% CIs.
    """

    df = df.copy()

    # Make sure offset is available and finite
    df = df[df['games'] > 0].copy()
    df['offset'] = np.log(df['games'].astype(float))

    # Formula using categorical controls for position and leagueCountry
    formula = (
        'redCards ~ SkinDark + meanIAT + meanExp + age + height + weight + '
        'C(position) + C(leagueCountry)'
    )

    # Fit negative binomial GLM with offset
    model = sm.GLM.from_formula(
        formula,
        data=df,
        family=sm.families.NegativeBinomial(),
        offset=df['offset']
    )

    res = model.fit()

    # Cluster robust covariance by referee id (refNum)
    try:
        res_clust = res.get_robustcov_results(cov_type='cluster', groups=df['refNum'])
    except Exception:
        # If clustering fails for some reason, fall back to default results
        res_clust = res

    # Compute incidence rate ratios (IRR = exp(coef)) and 95% CI
    params = res_clust.params
    conf = res_clust.conf_int()
    irr = np.exp(params)
    irr_ci_lower = np.exp(conf[0])
    irr_ci_upper = np.exp(conf[1])

    irr_table = pd.DataFrame({
        'IRR': irr,
        'CI_lower': irr_ci_lower,
        'CI_upper': irr_ci_upper
    })

    # Return both the fitted model (with clustered SEs) and the IRR table for interpretation
    return {
        'fitted_model_clustered': res_clust,
        'irr_table': irr_table
    }


