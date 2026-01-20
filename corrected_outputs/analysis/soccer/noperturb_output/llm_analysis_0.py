from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/noperturb_output/soccer.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dyad dataframe to the modeling dataframe.

    Produces these new/clean columns used in modeling:
      - SkinTone: mean of rater1 and rater2 (continuous 0-1)
      - SkinToneBin: binary indicator (1 dark, 0 light). Intermediate averages excluded.
      - PositionGroup: coarse position categories (Forward/Midfielder/Defender/Goalkeeper/Other)
      - logGames: natural log of 'games' used as offset/exposure in count models

    Drops rows missing the variables required to create the SkinTone binary classification
    or missing the redCards/games outcome.
    """
    df = df.copy()

    # Ensure required columns exist and drop rows with missing critical values
    required = [c for c in ['rater1', 'rater2', 'redCards', 'games'] if c in df.columns]
    df = df.dropna(subset=required)

    # Create continuous skin tone measure and binary indicator
    # If rater columns are present, compute mean
    if 'rater1' in df.columns and 'rater2' in df.columns:
        df['SkinTone'] = df[['rater1', 'rater2']].mean(axis=1)
        # Binarize: dark if >= 0.6, light if <= 0.4, intermediate set to NaN and removed
        df['SkinToneBin'] = np.where(df['SkinTone'] >= 0.6, 1,
                                     np.where(df['SkinTone'] <= 0.4, 0, np.nan))
        df = df[df['SkinToneBin'].notnull()].copy()
    else:
        # If raters missing, ensure SkinToneBin is absent so model will error later
        df['SkinTone'] = np.nan
        df['SkinToneBin'] = np.nan
        df = df[df['SkinToneBin'].notnull()].copy()  # will empty out

    # Coarse position grouping (keeps a small number of categories for dummies later)
    def _map_position(x):
        if not isinstance(x, str):
            return 'Other'
        s = x.lower()
        if 'midfield' in s:
            return 'Midfielder'
        if 'forward' in s or 'striker' in s or 'attacking' in s:
            return 'Forward'
        if 'defend' in s:
            return 'Defender'
        if 'goal' in s or 'keeper' in s:
            return 'Goalkeeper'
        return 'Other'

    if 'position' in df.columns:
        df['PositionGroup'] = df['position'].astype(str).apply(_map_position)
    else:
        df['PositionGroup'] = 'Other'

    # Ensure games is numeric and create log exposure
    if 'games' in df.columns:
        df['games'] = pd.to_numeric(df['games'], errors='coerce')
        df = df.dropna(subset=['games'])
        # games should be >= 1 by dataset description; nevertheless guard against zeros
        df = df[df['games'] > 0]
        df['logGames'] = np.log(df['games'])
    else:
        df['logGames'] = np.nan

    # Keep columns that will be used in modeling. If some of these are missing in the raw dataframe,
    # they will simply not be present in the returned dataframe (model code handles existence checks).
    keep_cols = [
        'playerShort', 'player', 'club', 'leagueCountry', 'position', 'PositionGroup', 'birthday',
        'height', 'weight', 'games', 'logGames', 'redCards', 'photoID', 'rater1', 'rater2',
        'SkinTone', 'SkinToneBin', 'refNum', 'meanIAT', 'meanExp'
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative binomial GLM for red card counts with exposure offset = log(games).

    Model form (in count-model notation):
      redCards ~ SkinToneBin + height + weight + meanIAT + meanExp + pos_dummies + league_dummies
    with offset = logGames and standard errors clustered at the referee level (refNum).

    Returns the fitted results object with cluster-robust covariance (attached when available).
    """
    df = df.copy()

    # Build dummy variables for categorical controls if present
    # PositionGroup -> pos_* dummies, drop_first=True to avoid multicollinearity with intercept
    if 'PositionGroup' in df.columns:
        pos_dummies = pd.get_dummies(df['PositionGroup'].astype(str), prefix='pos', drop_first=True)
        df = pd.concat([df, pos_dummies], axis=1)
    else:
        pos_dummies = pd.DataFrame(index=df.index)

    # leagueCountry -> league_* dummies
    if 'leagueCountry' in df.columns:
        league_dummies = pd.get_dummies(df['leagueCountry'].astype(str), prefix='league', drop_first=True)
        df = pd.concat([df, league_dummies], axis=1)
    else:
        league_dummies = pd.DataFrame(index=df.index)

    # Compose list of exogenous (control) variables to include
    base_controls = []
    for v in ['SkinToneBin', 'height', 'weight', 'meanIAT', 'meanExp']:
        if v in df.columns:
            base_controls.append(v)

    exog_vars = base_controls + list(pos_dummies.columns) + list(league_dummies.columns)

    if len(exog_vars) == 0:
        raise ValueError('No exogenous variables available for modeling. Check that required columns exist.')

    # Drop rows with missing values in exog, outcome, offset, or clustering var
    required_for_model = exog_vars + ['redCards', 'logGames', 'refNum']
    required_for_model = [c for c in required_for_model if c in df.columns]
    model_df = df.dropna(subset=required_for_model)

    # Final X and y
    X = model_df[exog_vars].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = model_df['redCards'].astype(float)
    offset = model_df['logGames'].astype(float)

    # Fit negative binomial; fall back to Poisson if NB fitting fails
    try:
        fam = sm.families.NegativeBinomial()
        glm = sm.GLM(y, X, family=fam, offset=offset)
        res = glm.fit()
    except Exception:
        glm = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
        res = glm.fit()

    # Compute cluster-robust SEs at referee level, if refNum exists
    clustered = res
    if 'refNum' in model_df.columns:
        # Preferred approach: use get_robustcov_results when available
        try:
            clustered = res.get_robustcov_results(cov_type='cluster', groups=model_df['refNum'])
        except Exception:
            # Fallback: compute clustered covariance matrix and attach to a lightweight wrapper
            try:
                from statsmodels.stats.sandwich_covariance import cov_cluster
                cov = cov_cluster(res, model_df['refNum'])
                class ClusteredResults:
                    def __init__(self, res_obj, cov_mat):
                        self._res = res_obj
                        self._cov = cov_mat
                        # set a bse attribute consistent with clustered covariance
                        try:
                            self.bse = np.sqrt(np.diag(cov_mat))
                        except Exception:
                            self.bse = getattr(res_obj, 'bse', None)

                    def cov_params(self):
                        return self._cov

                    def summary(self, *args, **kwargs):
                        # Return the original summary object (note: it will report default SEs).
                        # Users can access clustered covariance via cov_params() and clustered.bse.
                        return self._res.summary(*args, **kwargs)

                    def __getattr__(self, name):
                        return getattr(self._res, name)

                clustered = ClusteredResults(res, cov)
            except Exception:
                # If anything goes wrong, fall back to the original results (unclustered)
                clustered = res

    # Print brief summary and return results object with clustered cov if available
    try:
        print(clustered.summary())
    except Exception:
        try:
            print(res.summary())
        except Exception:
            pass

    return clustered