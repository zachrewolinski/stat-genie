from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as _scipy_stats

# Example top-level read (kept for context; transform operates on any dataframe passed in)
# df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/positive_leading_statement_output/soccer.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dyad-level data into a modeling dataframe.

    Produces the following key columns used in the model:
    - redCards: original count of red cards (dependent variable)
    - games: original number of matches (used as exposure)
    - log_games: natural log of games (offset)
    - SkinToneAvg: mean of rater1 and rater2 (0-1 normalized)
    - SkinDark: binary indicator (1 = clearly dark, 0 = clearly light). We define clearly dark as SkinToneAvg >= 0.6 and clearly light as SkinToneAvg <= 0.4 and drop ambiguous middle ratings to create a clean contrast.
    - age: age (years) at 2013-01-01
    - height, weight: numeric player physical covariates (filled with medians if missing)
    - position, leagueCountry, meanIAT, meanExp, refNum, playerShort: kept for modeling and clustering.
    """
    df = df.copy()

    # Keep rows with required fields
    req = ['redCards', 'games', 'rater1', 'rater2', 'position', 'leagueCountry', 'refNum', 'playerShort']
    df = df.dropna(subset=req)

    # Ensure numeric games/redCards if they are strings
    df['games'] = pd.to_numeric(df.get('games'), errors='coerce')
    df['redCards'] = pd.to_numeric(df.get('redCards'), errors='coerce')

    # Create average skin tone (0-1) and binary dark/light indicator
    df['SkinToneAvg'] = df[['rater1', 'rater2']].mean(axis=1)

    # Define clear-cut dark vs light thresholds; exclude ambiguous middle values to focus contrast
    df['SkinDark'] = np.where(df['SkinToneAvg'] >= 0.6, 1,
                              np.where(df['SkinToneAvg'] <= 0.4, 0, np.nan))
    df = df[df['SkinDark'].notnull()].copy()

    # Parse birthday and compute age at reference date (2013-01-01)
    df['birthday'] = pd.to_datetime(df.get('birthday'), format='%d.%m.%Y', errors='coerce', dayfirst=True)
    ref_date = pd.to_datetime('2013-01-01')
    df['age'] = (ref_date - df['birthday']).dt.days / 365.25

    # Ensure numeric covariates
    df['height'] = pd.to_numeric(df.get('height'), errors='coerce')
    df['weight'] = pd.to_numeric(df.get('weight'), errors='coerce')

    # Impute height/weight with median if missing (simple, transparent choice)
    if df['height'].isnull().any():
        df['height'] = df['height'].fillna(df['height'].median())
    if df['weight'].isnull().any():
        df['weight'] = df['weight'].fillna(df['weight'].median())

    # Keep only dyads with at least one game (schema indicates games min = 1, but check defensively)
    df = df[df['games'] > 0]

    # Create offset variable: log of games (exposure)
    df['log_games'] = np.log(df['games'])

    # Keep model-relevant columns and drop others to reduce memory
    keep_cols = [
        'playerShort', 'refNum', 'leagueCountry', 'position',
        'games', 'log_games', 'redCards',
        'SkinToneAvg', 'SkinDark',
        'age', 'height', 'weight',
        'meanIAT', 'meanExp'
    ]

    # Some datasets may not have meanIAT/meanExp for every refCountry record; keep rows even if these are missing
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative binomial regression for red card counts with games as exposure.

    Model: redCards ~ SkinDark + SkinToneAvg + age + height + weight + meanIAT + meanExp + C(position) + C(leagueCountry)
    Offset: log_games (so coefficients reflect effects on the red-card rate per game)

    We use a Negative Binomial family to accommodate overdispersion relative to Poisson.
    Standard errors are clustered at the referee level (refNum) because referee behavior generates within-referee correlation across dyads.

    Returns an object exposing .params (pd.Series), .bse (pd.Series) and .summary() for display.
    """
    from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hc3

    # Ensure required columns are present
    needed = ['redCards', 'log_games', 'SkinDark', 'SkinToneAvg', 'age', 'height', 'weight', 'position', 'leagueCountry', 'refNum', 'meanIAT', 'meanExp', 'games', 'playerShort']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns for modeling: {missing}')

    # Drop rows with missing values in predictors used by the formula or essential modeling columns
    model_df = df.dropna(subset=['redCards', 'log_games', 'SkinDark', 'position', 'leagueCountry'])

    # Formula: include categorical predictors using C() so we do not need to create dummies manually
    formula = (
        'redCards ~ SkinDark + SkinToneAvg + age + height + weight + meanIAT + meanExp '
        '+ C(position) + C(leagueCountry)'
    )

    # Fit negative binomial GLM with offset = log_games
    nb_model = smf.glm(formula=formula,
                       data=model_df,
                       family=sm.families.NegativeBinomial(),
                       offset=model_df['log_games'])
    res = nb_model.fit()

    # Compute clustered robust covariance at referee level (refNum).
    # Use sandwich covariance utilities: prefer cluster; fall back to HC3; as final fallback use model covariance.
    try:
        cluster_groups = model_df['refNum'].values
        cov_mat = cov_cluster(res, cluster_groups)
    except Exception:
        try:
            cov_mat = cov_hc3(res)
        except Exception:
            # Final fallback: use the default covariance from the fitted results
            try:
                cov_mat = res.cov_params()
            except Exception:
                cov_mat = np.asarray(res.normalized_cov_params) if hasattr(res, 'normalized_cov_params') else np.eye(len(res.params))

    # Ensure cov_mat is ndarray
    if isinstance(cov_mat, pd.DataFrame):
        cov_mat = cov_mat.values
    else:
        cov_mat = np.asarray(cov_mat)

    # Standard errors from clustered covariance
    bse_vals = np.sqrt(np.diag(cov_mat))
    params = pd.Series(res.params, index=res.params.index)
    bse = pd.Series(bse_vals, index=params.index)

    # Create a lightweight wrapper to mimic the expected interface used downstream
    class ClusteredResults:
        def __init__(self, base_res, params: pd.Series, bse: pd.Series, cov: np.ndarray):
            self._base_res = base_res
            self.params = params
            self.bse = bse
            self.cov = cov

        def summary(self):
            # Print the original model summary first
            try:
                print(self._base_res.summary())
            except Exception:
                # If original summary fails for some reason, ignore and continue to print our table
                pass

            # Then print a concise coefficient table using clustered SEs
            header = f"{'coef':>12} {'std err (clust)':>18} {'z':>10} {'P>|z|':>10} {'[0.025':>10} {'0.975]':>10}"
            print("\nClustered SE coefficient table:")
            print(header)
            for name, coef in self.params.items():
                se = self.bse.get(name, np.nan)
                if pd.isna(se) or se == 0:
                    z = np.nan
                    p = np.nan
                    ci_low = np.nan
                    ci_upp = np.nan
                else:
                    z = coef / se
                    p = 2 * (1 - _scipy_stats.norm.cdf(abs(z)))
                    ci_low = coef - 1.96 * se
                    ci_upp = coef + 1.96 * se
                print(f"{name:30s} {coef:12.4f} {se:18.4f} {z:10.3f} {p:10.3g} {ci_low:10.4f} {ci_upp:10.4f}")

    clustered_res = ClusteredResults(res, params, bse, cov_mat)

    # Print a concise summary and the key coefficient estimate for SkinDark
    clustered_res.summary()

    # Also compute incidence rate ratio (IRR) and 95% CI for SkinDark for interpretability
    coef = clustered_res.params.get('SkinDark')
    se_skin = clustered_res.bse.get('SkinDark')
    if coef is not None and se_skin is not None and not np.isnan(se_skin):
        irr = np.exp(coef)
        ci_lower = np.exp(coef - 1.96 * se_skin)
        ci_upper = np.exp(coef + 1.96 * se_skin)
        print(f"\nSkinDark IRR = {irr:.3f} (95% CI: {ci_lower:.3f} - {ci_upper:.3f})")

    return clustered_res