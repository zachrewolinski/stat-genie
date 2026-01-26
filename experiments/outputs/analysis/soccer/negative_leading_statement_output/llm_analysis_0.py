from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/negative_leading_statement_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dyad-level data into the modeling dataframe.
    Produces columns used in the model: redCards, games, log_games, SkinDark, SkinToneAvg,
    age, height, weight, position, leagueCountry, meanIAT, meanExp, refNum.

    Key decisions:
    - Use the mean of rater1 and rater2 as SkinToneAvg.
    - Restrict to extreme categories (dark vs light) to produce a clear binary comparison:
      SkinDark = 1 if SkinToneAvg >= 0.75 (top 2 categories), SkinDark = 0 if SkinToneAvg <= 0.25 (bottom 2 categories).
      Middle/neutral ratings are dropped to focus on the contrast of interest.
    - Use games as exposure via log_games offset (games >= 1 in dataset).
    - Parse birthday to compute age (reference date 2013-01-01 for the 2012-13 season).
    - Drop rows missing any variable required for the primary model.
    """
    # copy to avoid modifying in place
    df = df.copy()

    # Required raw columns for transformation
    required_cols = [
        'redCards', 'games', 'rater1', 'rater2', 'birthday',
        'height', 'weight', 'position', 'leagueCountry', 'meanIAT', 'meanExp', 'refNum'
    ]

    # Drop rows missing core required columns
    df = df.dropna(subset=required_cols)

    # Compute average skin tone from the two independent raters
    df['SkinToneAvg'] = df[['rater1', 'rater2']].mean(axis=1)

    # Restrict to extreme categories to compare dark vs light
    # Rater scores are normalized to [0,1] with 5 possible values -> typical values 0,0.25,0.5,0.75,1
    df['SkinDark'] = np.where(df['SkinToneAvg'] >= 0.75, 1,
                              np.where(df['SkinToneAvg'] <= 0.25, 0, np.nan))

    # Drop observations that are not in extreme categories (middle/neutral) or where SkinDark is NA
    df = df.dropna(subset=['SkinDark'])

    # Ensure games > 0 (dataset has min 1) and create log of games for offset
    df = df[df['games'] > 0]
    df['log_games'] = np.log(df['games'].astype(float))

    # Parse birthday (format dd.mm.yyyy in metadata); coerce errors to NaT and drop them
    df['birthday'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y', errors='coerce')
    df = df.dropna(subset=['birthday'])

    # Compute age at a reference date during the 2012-13 season (use 2013-01-01)
    ref_date = pd.to_datetime('2013-01-01')
    df['age'] = ((ref_date - df['birthday']).dt.days / 365.25).astype(float)

    # Ensure numeric height and weight
    df['height'] = pd.to_numeric(df['height'], errors='coerce')
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

    # Ensure categorical columns are of type category
    df['position'] = df['position'].astype('category')
    df['leagueCountry'] = df['leagueCountry'].astype('category')

    # Final drop of any rows missing modeling columns
    model_cols = [
        'redCards', 'games', 'log_games', 'SkinDark', 'SkinToneAvg', 'age',
        'height', 'weight', 'position', 'leagueCountry', 'meanIAT', 'meanExp', 'refNum'
    ]
    df = df.dropna(subset=model_cols)

    # Keep only columns necessary for modeling + a few informative columns
    keep_cols = model_cols + ['playerShort', 'player', 'club', 'photoID']
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a negative binomial regression for red card counts with games as exposure.

    Primary parameter of interest: coefficient on SkinDark (binary indicator for dark vs light skin tone).
    Model:
      redCards ~ SkinDark + age + height + weight + C(position) + C(leagueCountry) + meanIAT + meanExp
    Family: NegativeBinomial (GLM) with log link. Offset: log_games (log of number of games for the dyad).

    Robust (clustered) standard errors are computed at the referee (refNum) level.

    Returns a dictionary containing the fitted model, clustered results object, and a short summary of
    the SkinDark effect (coefficient, p-value, incidence rate ratio (IRR) and 95% CI).
    """
    import statsmodels.formula.api as smf

    # Formula: control for covariates and use categorical fixed effects for position and leagueCountry
    formula = (
        'redCards ~ SkinDark + age + height + weight + '
        'C(position) + C(leagueCountry) + meanIAT + meanExp'
    )

    # Fit GLM Negative Binomial with offset for exposure (log_games)
    model_glm = smf.glm(formula=formula,
                        data=df,
                        family=sm.families.NegativeBinomial(),
                        offset=df['log_games']).fit()

    # Compute cluster-robust covariance (cluster on referee ID)
    # Using statsmodels' cov_cluster to compute clustered covariance matrix and then
    # constructing a lightweight results-like object exposing params, pvalues, and conf_int().
    clustered_cov = cov_cluster(model_glm, df['refNum'])

    params = model_glm.params
    # Ensure cov shape aligns with params
    se_cluster = np.sqrt(np.diag(clustered_cov))
    # If for some reason covariance matrix has shape mismatch, fall back to model's bse
    if se_cluster.shape[0] != params.shape[0]:
        se_cluster = model_glm.bse.values

    z_scores = params.values / se_cluster
    pvalues = 2 * norm.sf(np.abs(z_scores))

    conf_lower = params.values - norm.ppf(0.975) * se_cluster
    conf_upper = params.values + norm.ppf(0.975) * se_cluster
    conf_df = pd.DataFrame({'lower': conf_lower, 'upper': conf_upper}, index=params.index)

    class ClusteredResults:
        def __init__(self, params_series: pd.Series, pvalues_array: np.ndarray, conf_dataframe: pd.DataFrame):
            self.params = params_series
            # store pvalues as a Series aligned with params index
            self.pvalues = pd.Series(pvalues_array, index=params_series.index)
            self._conf_df = conf_dataframe

        def conf_int(self):
            # Return the DataFrame with the same interface as statsmodels' conf_int()
            # (i.e., index matching param names and two columns)
            return self._conf_df

    clustered = ClusteredResults(params, pvalues, conf_df)

    # Extract SkinDark coefficient, p-value, and compute IRR and CI
    param_name = 'SkinDark'
    if param_name not in clustered.params.index:
        raise KeyError(f"Parameter {param_name} not in model. Available params: {clustered.params.index.tolist()}")

    coef = clustered.params[param_name]
    pval = float(clustered.pvalues[param_name])
    conf_int = clustered.conf_int().loc[param_name].tolist()  # [lower, upper]

    # Incidence Rate Ratio (IRR) and CI by exponentiating coefficients
    irr = float(np.exp(coef))
    irr_ci = [float(np.exp(conf_int[0])), float(np.exp(conf_int[1]))]

    results = {
        'glm_model': model_glm,
        'clustered_results': clustered,
        'SkinDark_coef': float(coef),
        'SkinDark_pvalue': float(pval),
        'SkinDark_IRR': irr,
        'SkinDark_IRR_CI': irr_ci,
        'model_formula': formula
    }

    return results