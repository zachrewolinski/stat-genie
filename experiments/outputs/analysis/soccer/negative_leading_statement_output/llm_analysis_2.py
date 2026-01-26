from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hc1
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/negative_leading_statement_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dyad-level data into analysis-ready dataframe.

    Produces the following new columns used in modeling:
      - meanSkin: mean of the two rater scores (0..1 normalized)
      - SkinCategory: categorical banding into Light/Medium/Dark
      - SkinDark: binary indicator (1 if Dark, 0 if Light)
      - age: age in years computed as of 2013-01-01
      - AnyRed: binary indicator whether redCards > 0
      - log_games: natural log of games (for offset)

    The function drops rows missing critical variables needed for the analysis.
    Only dyads where the player is coded as either Light or Dark are retained
    (we focus the main test on the contrast between light and dark skin tones).
    """

    df = df.copy()

    # Required columns for our analysis (inputs we try to preserve)
    required = ['rater1', 'rater2', 'redCards', 'games', 'birthday', 'position', 'leagueCountry', 'refNum']

    # Drop rows missing the truly essential fields
    df = df.dropna(subset=['rater1', 'rater2', 'redCards', 'games', 'birthday'])

    # Compute mean skin rating from the two raters (the dataset provides normalized rater values 0..1)
    df['meanSkin'] = (df['rater1'] + df['rater2']) / 2.0

    # Band into Light / Medium / Dark using the natural thirds of the normalized 0..1 scale
    def skin_cat(x):
        if x <= 0.3333333:
            return 'Light'
        elif x >= 0.6666667:
            return 'Dark'
        else:
            return 'Medium'

    df['SkinCategory'] = df['meanSkin'].apply(skin_cat)

    # Keep only Light vs Dark to make the direct contrast requested in the research question
    df = df[df['SkinCategory'].isin(['Light', 'Dark'])].copy()

    # Binary indicator: 1 if Dark, 0 if Light
    df['SkinDark'] = (df['SkinCategory'] == 'Dark').astype(int)

    # Parse birthday and compute age as of 2013-01-01 (mid-season reference)
    df['birthday'] = pd.to_datetime(df['birthday'], dayfirst=True, errors='coerce')
    reference_date = pd.to_datetime('2013-01-01')
    df['age'] = ((reference_date - df['birthday']).dt.days / 365.25).astype(float)

    # Binary outcome for robustness: whether any red card was issued by this referee to this player
    df['AnyRed'] = (df['redCards'] > 0).astype(int)

    # Ensure games is positive (cannot take log of zero); drop dyads with zero games
    df = df[df['games'] > 0].copy()

    # Log offset for rate models
    df['log_games'] = np.log(df['games'].astype(float))

    # Keep only columns necessary for modeling plus a few informative columns
    keep_cols = [
        'playerShort', 'player', 'photoID', 'position', 'club', 'leagueCountry',
        'birthday', 'age', 'height', 'weight', 'games', 'log_games',
        'redCards', 'AnyRed', 'rater1', 'rater2', 'meanSkin', 'SkinCategory', 'SkinDark',
        'refNum', 'refCountry', 'meanIAT', 'meanExp', 'nIAT', 'nExp'
    ]

    # Some columns may not exist in all dataset exports; intersect to avoid KeyError
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    # Ensure that the FINAL dataframe contains the conceptual variables (columns) required by the model.
    # If any are missing from the dataset, add them as NaN so downstream code can run and will drop rows as needed.
    final_required_cols = [
        'SkinDark',    # IV
        'redCards', 'AnyRed',  # DVs
        'age', 'height', 'weight', 'position', 'leagueCountry', 'meanIAT', 'meanExp', 'games', 'log_games', 'refNum'
    ]
    for col in final_required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Keep columns in a stable order (optional)
    # Ensure we don't drop any of the required columns
    final_cols_order = [c for c in [
        'playerShort', 'player', 'photoID', 'position', 'club', 'leagueCountry',
        'birthday', 'age', 'height', 'weight', 'games', 'log_games',
        'redCards', 'AnyRed', 'rater1', 'rater2', 'meanSkin', 'SkinCategory', 'SkinDark',
        'refNum', 'refCountry', 'meanIAT', 'meanExp', 'nIAT', 'nExp'
    ] if c in df.columns]
    df = df[final_cols_order].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Runs two complementary models to test whether dark-skinned players receive more red cards
    than light-skinned players in player-referee dyads.

    1) Primary model: Negative binomial GLM for redCards with offset=log_games to model red-card rate
       (red cards per game). Coefficients are exponentiated to produce incidence rate ratios (IRRs).
       Standard errors are clustered at the referee (refNum) level.

    2) Robustness model: Logistic (binomial) GLM predicting whether the dyad had any red card
       (AnyRed). Standard errors likewise clustered at refNum.

    The function returns a dictionary with fitted model objects (raw fit objects) and IRRs computed
    using cluster-robust covariance matrices (when possible).
    """

    # Drop rows with missing values in model-important columns (these columns are required conceptual vars)
    required_for_model = ['redCards', 'games', 'log_games', 'SkinDark', 'age', 'height', 'weight', 'position', 'leagueCountry', 'refNum', 'meanIAT', 'meanExp']
    # Only include those that actually exist in the dataframe to avoid KeyError (transform ensures presence but may be NaN)
    required_for_model = [c for c in required_for_model if c in df.columns]
    model_df = df.dropna(subset=required_for_model).copy()

    # Formula: skin main effect + plausible player-level controls + country-level bias controls + categorical controls
    formula = 'redCards ~ SkinDark + age + height + weight + meanIAT + meanExp + C(position) + C(leagueCountry)'

    # Fit negative binomial GLM with offset = log_games
    nb_model_raw = smf.glm(formula=formula, data=model_df, family=sm.families.NegativeBinomial(), offset=model_df['log_games']).fit()

    # Attempt to compute cluster-robust covariance matrix clustered by referee ID; fall back to HC1 if cluster fails
    try:
        nb_cov = cov_cluster(nb_model_raw, model_df['refNum'])
    except Exception:
        nb_cov = cov_hc1(nb_model_raw)

    # Compute incidence rate ratios (IRR) and 95% CIs from the clustered NB covariance
    nb_params = nb_model_raw.params
    nb_se_cluster = np.sqrt(np.diag(nb_cov))
    irr = np.exp(nb_params)
    z = 1.96
    irr_ci_lower = np.exp(nb_params - z * nb_se_cluster)
    irr_ci_upper = np.exp(nb_params + z * nb_se_cluster)

    # Robustness: predict AnyRed using binomial family (logistic). We include same covariates.
    formula_bin = 'AnyRed ~ SkinDark + age + height + weight + meanIAT + meanExp + C(position) + C(leagueCountry)'
    bin_model_raw = smf.glm(formula=formula_bin, data=model_df, family=sm.families.Binomial()).fit()

    try:
        bin_cov = cov_cluster(bin_model_raw, model_df['refNum'])
    except Exception:
        bin_cov = cov_hc1(bin_model_raw)

    # Print brief summaries for interactive inspection
    print('=== Negative Binomial (rate) model (raw fit) ===')
    print(nb_model_raw.summary())
    print('\nIncidence rate ratios (IRR) and 95% CI (cluster-robust where available):')
    irr_table = pd.DataFrame({'IRR': irr, 'IRR_CI_lower': irr_ci_lower, 'IRR_CI_upper': irr_ci_upper})
    print(irr_table)

    print('\n=== Binomial (AnyRed) model (raw fit) ===')
    print(bin_model_raw.summary())

    # Return key objects for programmatic inspection. We include the raw fitted result objects and the clustered covariances.
    return {
        'nb_model_clustered': nb_model_raw,         # raw fitted result; cluster cov is provided separately
        'nb_cluster_cov': nb_cov,                   # clustered covariance matrix (or HC1 fallback)
        'nb_cluster_bse': nb_se_cluster,            # clustered standard errors
        'nb_irr_table': irr_table,
        'binomial_model_clustered': bin_model_raw,  # raw fitted logistic result
        'bin_cluster_cov': bin_cov,
        'model_df_rows': len(model_df)
    }