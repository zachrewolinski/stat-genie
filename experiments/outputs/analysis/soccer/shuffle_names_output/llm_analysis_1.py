from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hc1
from scipy import stats

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/shuffle_names_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis-ready dataframe. Key steps:
    - Compute mean skin tone from rater1 and rater2
    - Create categorical skin tone (Dark / Light / Ambiguous) and restrict to Dark vs Light
    - Create binary DarkSkin indicator
    - Ensure red-card count and games (exposure) are numeric and remove invalid rows
    - Bring control variables into numeric form (leagueCountry and club) and ensure referee/player ids are numeric

    Columns produced and required for modeling:
    - DarkSkin (0/1)
    - red_card_count (count outcome)
    - exposure_games (exposure; number of games in dyad)
    - leagueCountry (numeric; implicit bias measure for referee country)
    - club (numeric; explicit bias measure for referee country)
    - goals (referee id; used for clustering)
    - player (player id)
    """
    df = df.copy()

    # Convert rater columns to numeric and compute mean skin tone
    df['rater1'] = pd.to_numeric(df.get('rater1'), errors='coerce')
    df['rater2'] = pd.to_numeric(df.get('rater2'), errors='coerce')
    df['SkinToneMean'] = df[['rater1', 'rater2']].mean(axis=1)

    # Create categorical variable: treat values >=0.6 as Dark, <=0.4 as Light, middle as Ambiguous
    # (Scale: raters are normalized to 0..1 across 5-level scale per dataset description)
    df['SkinToneCategory'] = pd.cut(df['SkinToneMean'], bins=[-0.01, 0.4, 0.6, 1.01], labels=['Light', 'Ambiguous', 'Dark'])

    # Keep only clearly Dark or Light (exclude Ambiguous/middle) to focus the comparison
    df = df[df['SkinToneCategory'].isin(['Dark', 'Light'])].copy()

    # Binary indicator: 1 = Dark skin, 0 = Light skin
    df['DarkSkin'] = (df['SkinToneCategory'] == 'Dark').astype(int)

    # Outcome: red card count. According to available fields, 'photoID' carries small integer counts (0-2)
    # which corresponds to number of red cards from the focal referee in the dyad.
    df['red_card_count'] = pd.to_numeric(df.get('photoID'), errors='coerce')

    # Exposure: number of games in the player-referee dyad. Use 'games' (must be > 0)
    df['exposure_games'] = pd.to_numeric(df.get('games'), errors='coerce')

    # Drop rows with missing or invalid outcome/exposure/IV
    df = df.dropna(subset=['red_card_count', 'exposure_games', 'DarkSkin'])
    df = df[df['exposure_games'] > 0]

    # Controls: numeric implicit/explicit bias measures for referee country
    df['leagueCountry'] = pd.to_numeric(df.get('leagueCountry'), errors='coerce')
    df['club'] = pd.to_numeric(df.get('club'), errors='coerce')

    # Referee and player identifiers (ensure numeric for clustering/grouping)
    # According to the dataset descriptions, 'goals' corresponds to referee id and 'player' to player id.
    df['goals'] = pd.to_numeric(df.get('goals'), errors='coerce')
    df['player'] = pd.to_numeric(df.get('player'), errors='coerce')

    # Drop rows missing key controls (you might relax this if you want to keep rows with missing country-level bias)
    df = df.dropna(subset=['leagueCountry', 'club', 'goals', 'player'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a count model for red card counts with exposure (number of games).

    Primary specification:
    - Negative binomial GLM with log(exposure_games) as an offset
    - Key coefficient: DarkSkin (whether player has dark skin tone)
    - Controls: leagueCountry (implicit bias), club (explicit bias)
    - Cluster-robust standard errors at the referee level (goals)

    Returns a dictionary with the fitted model object (robust), IRRs (incidence rate ratios), CIs and textual summary.
    """
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hc1
    from scipy import stats

    df = df.copy()

    # Prepare regressors
    exog_vars = ['DarkSkin', 'leagueCountry', 'club']
    exog = df[exog_vars].copy()
    exog = sm.add_constant(exog)

    # Offset = log(number of games / exposure)
    offset = np.log(df['exposure_games'].astype(float))

    # Fit Negative Binomial GLM (accounts for overdispersion relative to Poisson)
    model_glm = sm.GLM(df['red_card_count'].astype(float), exog, family=sm.families.NegativeBinomial(), offset=offset)
    res = model_glm.fit()

    # Obtain cluster-robust covariance matrix clustered by referee id (goals). If clustering fails, fallback to HC1.
    try:
        clustered_cov = cov_cluster(res, df['goals'])
    except Exception:
        clustered_cov = cov_hc1(res)

    # Params and robust confidence intervals (using normal approximation, 95% CI)
    params = res.params
    se_robust = np.sqrt(np.diag(clustered_cov))
    z = stats.norm.ppf(0.975)
    conf_lower = params - z * se_robust
    conf_upper = params + z * se_robust
    conf_df = pd.DataFrame({0: conf_lower, 1: conf_upper})

    # Incidence rate ratios and confidence intervals
    irr = np.exp(params)
    irr_ci = np.exp(conf_df)

    out = {
        'model_result': res,  # original fitted result object
        'summary_text': res.summary().as_text(),
        'incidence_rate_ratios': irr,
        'irr_conf_int': irr_ci,
        'formula': 'red_card_count ~ DarkSkin + leagueCountry + club   (offset = log(exposure_games))',
        'exog_vars': exog_vars,
        'offset_var': 'exposure_games',
        'cluster_var': 'goals'
    }

    return out