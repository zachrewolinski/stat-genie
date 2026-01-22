from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/negative_leading_statement_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dyad dataframe into an analysis-ready dataset for a count model.

    Creates the following columns (exact names used later in the model):
      - SkinAvg : average of rater1 and rater2 ratings
      - SkinDark: binary 1 if average rating in the top skin-tone categories (dark), 0 if in the bottom (light). We filter to extreme-rated photos only (dark vs light) to test the direct contrast.
      - Age: player's age in years (at season midpoint 2013-01-01)
      - log_games: natural log of 'games' used as offset (exposure). 'games' are required to be positive.

    Drops rows with missing critical fields.
    """
    df = df.copy()

    # Keep only rows with the columns we need
    required = ['rater1', 'rater2', 'redCards', 'games', 'birthday', 'height', 'weight', 'meanIAT', 'meanExp', 'refNum']
    df = df.dropna(subset=required)

    # Compute mean rater score
    df['SkinAvg'] = (df['rater1'].astype(float) + df['rater2'].astype(float)) / 2.0

    # Create binary dark vs light groups: use extremes to make the comparison clean.
    # Rater scale in data appears normalized to 0..1 with 5 discrete values (0, 0.25, 0.5, 0.75, 1).
    # Define dark as average >= 0.75, light as average <= 0.25. Drop intermediate (mixed/medium) images.
    df = df[df['SkinAvg'].notnull()]
    df = df[(df['SkinAvg'] >= 0.75) | (df['SkinAvg'] <= 0.25)].copy()
    df['SkinDark'] = (df['SkinAvg'] >= 0.75).astype(int)

    # Ensure games is positive (schema indicates min 1). Compute offset
    df = df[df['games'] > 0].copy()
    df['log_games'] = np.log(df['games'].astype(float))

    # Parse birthday and compute age at a reference date (season midpoint)
    # birthday format is dd.mm.yyyy according to schema
    def safe_parse_date(col):
        return pd.to_datetime(col, dayfirst=True, errors='coerce', format='%d.%m.%Y')

    df['birthday_parsed'] = safe_parse_date(df['birthday'])
    season_mid = pd.to_datetime('2013-01-01')
    df['Age'] = (season_mid - df['birthday_parsed']).dt.days / 365.25

    # Drop remaining rows without parsed birthday or any critical numeric columns
    df = df.dropna(subset=['Age', 'height', 'weight', 'meanIAT', 'meanExp'])

    # Keep only columns necessary for the modeling (plus a few identifiers)
    keep_cols = [
        'playerShort', 'refNum', 'refCountry',
        'redCards', 'games', 'log_games',
        'SkinAvg', 'SkinDark',
        'Age', 'height', 'weight',
        'meanIAT', 'meanExp'
    ]

    # Some columns may not exist in extremely sparse or malformed inputs; guard against that
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a negative binomial regression for counts of red cards with exposure = games (offset = log_games).

    Model specification:
      redCards ~ SkinDark + Age + height + weight + meanIAT + meanExp
      offset = log_games

    We estimate robust (clustered) standard errors at the referee level (refNum) to account for dyad-level dependence by referee.

    Returns the fitted model results (statsmodels results object) and prints a brief summary.
    """
    # Build design matrices
    features = ['SkinDark', 'Age', 'height', 'weight', 'meanIAT', 'meanExp']
    for f in features:
        if f not in df.columns:
            raise ValueError(f"Required feature column missing from dataframe: {f}")

    X = df[features].astype(float)
    X = sm.add_constant(X, has_constant='add')
    y = df['redCards'].astype(float)
    offset = df['log_games'].astype(float)

    # Fit GLM Negative Binomial with offset; cluster robust SEs by refNum
    model_glm = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)

    # Fit with cluster-robust covariances grouped by refNum
    try:
        res = model_glm.fit(cov_type='cluster', cov_kwds={'groups': df['refNum'].values})
    except Exception:
        # fallback: fit without clustered covariance if that fails, but still return the results
        res = model_glm.fit()

    # Print summary for quick inspection
    print(res.summary())

    # Also calculate incidence rate ratios (IRRs) and 95% CI for interpretability
    params = res.params
    conf = res.conf_int()
    irrs = np.exp(params)
    irrs_ci_lower = np.exp(conf[0])
    irrs_ci_upper = np.exp(conf[1])
    irr_table = pd.DataFrame({
        'IRR': irrs,
        'CI_lower': irrs_ci_lower,
        'CI_upper': irrs_ci_upper
    })
    print('\nIncidence Rate Ratios (IRRs) with 95% CI:')
    print(irr_table.loc[features + ['const']])

    return {
        'results': res,
        'irr_table': irr_table
    }


