from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hc3
from scipy import stats as sps

# Read input CSV (kept from original context; transform() can accept other DataFrames too)
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/noperturb_output/soccer.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dyad dataframe into the analysis-ready dataframe.

    Steps performed:
    - Parse birthday into datetime and compute age (reference year 2013, season 2012-2013).
    - Compute average skin tone from the two raters and create a 3-category SkinToneCat (Light, Other, Dark).
      Keep only dyads where players are clearly Light or Dark (to directly answer the research Q).
    - Create binary Dark indicator (1 = Dark, 0 = Light).
    - Remove rows with missing values in key analysis columns and require games>0.
    - Create exposure offset log_games = log(games).
    - Compute per-game controls: goals_per_game, yellowCards_per_game.
    - Standardize continuous controls (z-scores) used in the model.

    Returns the dataframe with the columns listed in the conceptual variables.
    """

    df = df.copy()

    # Parse birthday (format dd.mm.yyyy) and compute age at season reference (2013)
    df['birthday_dt'] = pd.to_datetime(df.get('birthday'), format='%d.%m.%Y', errors='coerce')
    # If parse fails, leave NaT; age will be NaN and the row will be dropped later
    df['age'] = 2013 - df['birthday_dt'].dt.year

    # Compute average skin rating from two raters
    df['SkinToneAvg'] = df[['rater1', 'rater2']].mean(axis=1)

    # Categorize: Light if avg <= 0.25 (1 or 2 on 5-point scale), Dark if avg >= 0.75 (4 or 5), else Other
    def skin_cat(x):
        if pd.isna(x):
            return pd.NA
        if x <= 0.25:
            return 'Light'
        if x >= 0.75:
            return 'Dark'
        return 'Other'

    df['SkinToneCat'] = df['SkinToneAvg'].apply(skin_cat)

    # Keep only clearly Light or Dark players (research question compares Dark vs Light)
    df = df[df['SkinToneCat'].isin(['Light', 'Dark'])]

    # Create binary Dark indicator: 1 dark, 0 light
    df['Dark'] = (df['SkinToneCat'] == 'Dark').astype(int)

    # Ensure games > 0 (need exposure) and drop rows with missing in key columns
    required_cols = ['redCards', 'games', 'rater1', 'rater2', 'refNum', 'meanIAT', 'meanExp', 'position', 'leagueCountry']
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    # Ensure numeric games and positive
    if 'games' in df.columns:
        df['games'] = pd.to_numeric(df['games'], errors='coerce')
        df = df[df['games'] > 0]
    else:
        # If games column is not present, resulting dataset cannot be used; let subsequent checks handle it
        pass

    # Exposure offset: log(games)
    df['log_games'] = np.log(df['games'])

    # Per-game measures (guard against division by zero - games already filtered to >0)
    df['goals_per_game'] = np.nan
    df['yellowCards_per_game'] = np.nan
    if 'goals' in df.columns:
        df['goals_per_game'] = df['goals'] / df['games']
    if 'yellowCards' in df.columns:
        df['yellowCards_per_game'] = df['yellowCards'] / df['games']

    # Replace infinite / very large values (if any) with NaN and drop later
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Standardize continuous controls (z-scores). Use mean/std computed on the kept sample.
    def zscore(col: pd.Series) -> pd.Series:
        col_clean = col.dropna()
        std = col_clean.std(ddof=0)
        if std == 0 or np.isnan(std):
            std = 1.0
        return (col - col.mean()) / std

    # compute standardized controls only if columns exist; otherwise they'll be absent and model() will catch
    if 'meanIAT' in df.columns:
        df['meanIAT_z'] = zscore(df['meanIAT'])
    if 'meanExp' in df.columns:
        df['meanExp_z'] = zscore(df['meanExp'])
    if 'age' in df.columns:
        df['age_z'] = zscore(df['age'])
    if 'height' in df.columns:
        df['height_z'] = zscore(df['height'])
    if 'weight' in df.columns:
        df['weight_z'] = zscore(df['weight'])
    # Fill NaNs for per-game before z-scoring to avoid dropping too many rows; treat missing as 0 performance in dyad
    df['goals_per_game_z'] = zscore(df['goals_per_game'].fillna(0)) if 'goals_per_game' in df.columns else None
    df['yellowCards_per_game_z'] = zscore(df['yellowCards_per_game'].fillna(0)) if 'yellowCards_per_game' in df.columns else None

    # Keep only columns necessary for modeling and diagnostics
    keep_cols = [
        'playerShort', 'player', 'club', 'leagueCountry', 'position', 'birthday_dt', 'age',
        'height', 'weight', 'games', 'log_games', 'goals', 'goals_per_game', 'yellowCards', 'yellowCards_per_game',
        'redCards', 'photoID', 'rater1', 'rater2', 'SkinToneAvg', 'SkinToneCat', 'Dark',
        'refNum', 'refCountry', 'meanIAT', 'meanIAT_z', 'nIAT', 'seIAT', 'meanExp', 'meanExp_z', 'nExp', 'seExp',
        'goals_per_game_z', 'yellowCards_per_game_z', 'age_z', 'height_z', 'weight_z'
    ]

    # Some columns may be missing in input; intersect with available columns
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Final drop of any rows with NaNs in model columns
    model_cols = ['redCards', 'log_games', 'Dark', 'meanIAT_z', 'meanExp_z', 'age_z', 'height_z', 'weight_z', 'goals_per_game_z', 'yellowCards_per_game_z', 'position', 'leagueCountry', 'refNum']
    model_cols = [c for c in model_cols if c in df.columns]
    df = df.dropna(subset=model_cols)

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative binomial regression of red card counts on player skin tone (Dark vs Light),
    controlling for covariates and using log(games) as an exposure offset. Cluster standard errors at the referee level (refNum).

    Returns the robust model results object (robust covariance clustered by referee) and a text summary using clustered SEs.
    """

    # Ensure necessary columns exist
    required = ['redCards', 'log_games', 'Dark', 'meanIAT_z', 'meanExp_z', 'age_z', 'height_z', 'weight_z',
                'goals_per_game_z', 'yellowCards_per_game_z', 'position', 'leagueCountry', 'refNum']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for modeling: {missing}")

    # Build formula. Use C(...) for categorical covariates.
    formula = (
        'redCards ~ Dark + meanIAT_z + meanExp_z + age_z + height_z + weight_z '
        '+ goals_per_game_z + yellowCards_per_game_z + C(position) + C(leagueCountry)'
    )

    # Fit GLM negative binomial with offset = log_games
    model_glm = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=df['log_games'])
    res = model_glm.fit()

    # Obtain cluster-robust covariance clustered by refNum (referee)
    # Use sandwich estimator to compute clustered covariance matrix and build a lightweight wrapper with clustered SEs
    try:
        cluster_groups = df['refNum'].values
        cov = cov_cluster(res, cluster_groups)
    except Exception:
        # Fallback to HC3 if clustering fails
        cov = cov_hc3(res)

    # Compute clustered standard errors, t-stats, and p-values (using normal approximation)
    params = res.params
    bse = np.sqrt(np.diag(cov))
    # Guard against zero division
    bse_safe = np.where(bse == 0, np.nan, bse)
    tvalues = params / bse_safe
    pvalues = 2 * (1 - sps.norm.cdf(np.abs(tvalues)))

    # Minimal results wrapper for clustered results
    class ClusteredResults:
        def __init__(self, orig_res, cov_matrix, params, bse, tvalues, pvalues):
            self.orig_res = orig_res
            self._cov = cov_matrix
            self.params = params
            self.bse = bse
            self.tvalues = tvalues
            self.pvalues = pvalues

        def cov_params(self):
            return self._cov

        def summary(self) -> str:
            # Build a simple coefficient table summary using clustered SEs
            coef_df = pd.DataFrame({
                'coef': self.params,
                'std_err': self.bse,
                't': self.tvalues,
                'P>|z|': self.pvalues,
                '[0.025': self.params - 1.96 * self.bse,
                '0.975]': self.params + 1.96 * self.bse
            })
            # Include model family and basic fit info from original result
            header = []
            try:
                header.append(f"Model: {self.orig_res.model.__class__.__name__}")
                header.append(f"Family: {getattr(self.orig_res.model, 'family', None)}")
                header.append(f"Method: {getattr(self.orig_res, 'method', 'fit')}")
                header.append(f"Number of observations: {int(self.orig_res.nobs)}")
            except Exception:
                pass
            header_text = "\n".join(header)
            return header_text + "\n\n" + coef_df.to_string()

    res_clust = ClusteredResults(res, cov, params, bse, tvalues, pvalues)

    # For convenience return both the original fit and the clustered-robust result
    # summary is provided as the clustered summary text
    return {
        'model': model_glm,
        'result': res,
        'result_clustered': res_clust,
        'summary': res_clust.summary()
    }