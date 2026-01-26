from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy import stats


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dyad dataframe into the analysis dataframe.
    Steps:
    - Compute skin_score as the mean of rater1 and rater2 (they are normalized to 0-1).
    - Create binary dark/light indicators and keep only clearly dark or clearly light observations (drops middle category) so the primary test compares Dark vs Light.
    - Parse birthday to compute age (season year assumed 2013).
    - Ensure games>0 and compute offset = log(games).
    - Coerce numeric controls to numeric and drop rows missing required fields for the primary analysis.
    Returned dataframe contains the exact column names used in the model.
    """

    df = df.copy()

    # Convert rater columns to numeric (they are normalized 0-1 but may contain strings/NaN)
    df['rater1'] = pd.to_numeric(df.get('rater1'), errors='coerce')
    df['rater2'] = pd.to_numeric(df.get('rater2'), errors='coerce')

    # Compute average skin score (range 0-1)
    df['skin_score'] = df[['rater1', 'rater2']].mean(axis=1)

    # Drop rows without any skin rating or without outcome/exposure/ref id
    df = df.dropna(subset=['skin_score', 'redCards', 'games', 'refNum'])

    # Define dark vs light thresholds on normalized 0-1 scale.
    # These thresholds approximate dark (>=0.6) vs light (<=0.4). Middle values are considered 'medium' and excluded
    # to make the primary comparison a clean dark vs light test.
    df['skin_binary_dark'] = (df['skin_score'] >= 0.6).astype(int)
    df['skin_binary_light'] = (df['skin_score'] <= 0.4).astype(int)

    # Keep only clearly dark or clearly light players for main test
    df = df[(df['skin_binary_dark'] == 1) | (df['skin_binary_light'] == 1)].copy()

    # Parse birthday and compute age (season approximated as 2013)
    # birthday format in schema is dd.mm.yyyy
    df['birthday'] = pd.to_datetime(df.get('birthday'), format='%d.%m.%Y', errors='coerce')
    df['age'] = 2013 - df['birthday'].dt.year

    # Keep only dyads with at least one game; compute offset (log exposure)
    df = df[df['games'] > 0].copy()
    df['offset'] = np.log(df['games'])

    # Ensure numeric controls
    df['height'] = pd.to_numeric(df.get('height'), errors='coerce')
    df['weight'] = pd.to_numeric(df.get('weight'), errors='coerce')
    df['meanIAT'] = pd.to_numeric(df.get('meanIAT'), errors='coerce')
    df['meanExp'] = pd.to_numeric(df.get('meanExp'), errors='coerce')

    # Position and leagueCountry should be present; coerce to string categories if present
    if 'position' in df.columns:
        df['position'] = df['position'].astype('category')
    if 'leagueCountry' in df.columns:
        df['leagueCountry'] = df['leagueCountry'].astype('category')

    # Drop rows missing essential controls needed in the primary specification
    required_controls = ['age', 'height', 'weight', 'position', 'leagueCountry', 'meanIAT', 'meanExp']
    # Only include those required controls that are present in df to avoid KeyError; later failing if missing columns is expected behavior.
    required_present = [c for c in required_controls if c in df.columns]
    df = df.dropna(subset=required_present)

    # Final selection: keep columns used in modeling plus identifiers
    keep_cols = [
        'playerShort', 'refNum', 'redCards', 'games', 'offset',
        'skin_score', 'skin_binary_dark',
        'age', 'position', 'leagueCountry', 'height', 'weight',
        'meanIAT', 'meanExp'
    ]
    # Keep only the existing ones (defensive) and return
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


def model(df: pd.DataFrame) -> Any:
    """
    Fit primary negative binomial GLM for red card counts with log(games) as offset.
    - Primary test: coefficient on `skin_binary_dark` (dark = 1) tests whether darker players receive more red cards per game than lighter players.
    - Continuous sensitivity: `skin_score` included as continuous check.
    - Controls: age, height, weight, categorical position, categorical leagueCountry, and referee-country bias measures (meanIAT, meanExp).
    - Cluster-robust standard errors by referee id (refNum) to account for within-referee correlation.
    Returns a dictionary containing the main model, clustered-robust results (wrapped), a Poisson sensitivity model, and an overdispersion statistic.
    """

    # copy to avoid side-effects
    df = df.copy()

    # Primary formula: redCards as function of skin darkness (binary), continuous skin_score, controls and categorical covariates
    formula = (
        'redCards ~ skin_binary_dark + skin_score + age + height + weight '
        '+ C(position) + C(leagueCountry) + meanIAT + meanExp'
    )

    # Negative Binomial GLM (models counts with overdispersion) using offset = log(games)
    nb_model = smf.glm(formula=formula, data=df,
                      family=sm.families.NegativeBinomial(),
                      offset=df['offset']).fit()

    # Attempt to compute cluster-robust covariance matrix by referee id
    # statsmodels' GLMResults may not always provide get_robustcov_results; compute clustered cov explicitly
    clustered_cov = cov_cluster(nb_model, df['refNum'])

    # Create a simple wrapper object exposing common attributes (params, bse, pvalues, cov_params, tvalues)
    class ClusteredResults:
        def __init__(self, base_res, cov):
            self.model = base_res
            self.params = base_res.params
            self.cov_params = cov
            self.bse = np.sqrt(np.diag(cov))
            # Use normal approximation for z-stats and p-values
            self.tvalues = self.params / self.bse
            self.pvalues = 2 * (1 - stats.norm.cdf(np.abs(self.tvalues)))

        def summary(self):
            # Return the base model summary with a note that standard errors are clustered (users can inspect cov_params/bse)
            return self.model.summary()

    nb_clustered = ClusteredResults(nb_model, clustered_cov)

    # Poisson model as a sensitivity check (will typically underestimate SEs if overdispersion present)
    poisson_model = smf.glm(formula=formula, data=df,
                           family=sm.families.Poisson(),
                           offset=df['offset']).fit()

    # Overdispersion check (ratio of deviance to residual df) -- values >> 1 indicate overdispersion
    overdispersion = float(nb_model.deviance / nb_model.df_resid)

    # Return the fitted objects and summary info.
    return {
        'nb_model': nb_model,
        'nb_clustered': nb_clustered,
        'poisson_model': poisson_model,
        'overdispersion': overdispersion
    }