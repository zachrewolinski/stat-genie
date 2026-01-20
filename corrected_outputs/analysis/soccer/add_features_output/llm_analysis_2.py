from typing import Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/add_features_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Keep rows with required variables present. We need rater1 and rater2 (photo-based skin ratings),
    # redCards (count outcome), games (exposure), and country-level bias measures.
    required = ['redCards', 'games', 'rater1', 'rater2', 'meanIAT', 'meanExp', 'position', 'leagueCountry', 'refNum']
    df = df.dropna(subset=required)

    # Create averaged skin rating from the two independent raters
    df['SkinAvg'] = df[['rater1', 'rater2']].mean(axis=1)

    # Define clear Dark vs Light groups to maximize discrimination.
    # Dark: SkinAvg >= 0.60; Light: SkinAvg <= 0.40. Drop middle/ambiguous cases (0.40 < SkinAvg < 0.60).
    df = df[df['SkinAvg'].notnull()]
    df = df[(df['SkinAvg'] <= 0.40) | (df['SkinAvg'] >= 0.60)].copy()
    df['SkinDark'] = (df['SkinAvg'] >= 0.60).astype(int)

    # Parse birthday and create Age in years (reference year 2013, mid-season). If parsing fails, set NA.
    df['birthday_parsed'] = pd.to_datetime(df.get('birthday', pd.Series(dtype='str')), format='%d.%m.%Y', errors='coerce')
    df['Age'] = 2013 - df['birthday_parsed'].dt.year

    # Ensure numeric height/weight
    df['height'] = pd.to_numeric(df.get('height', pd.Series(dtype='float')), errors='coerce')
    df['weight'] = pd.to_numeric(df.get('weight', pd.Series(dtype='float')), errors='coerce')

    # Drop rows missing the numeric controls we will use
    df = df.dropna(subset=['Age', 'height', 'weight'])

    # Exposure offset for count model: log(games). games should be >= 1 in this dataset, but guard anyway.
    # If games <= 0 (shouldn't occur), set to NaN so they are dropped.
    df['games'] = pd.to_numeric(df['games'], errors='coerce')
    df = df[df['games'] > 0]
    df['offset'] = np.log(df['games'])

    # Keep a reduced set of columns needed for modeling and identifiers for diagnostics
    keep_cols = [
        'playerShort', 'player', 'refNum', 'refCountry',
        'games', 'redCards', 'SkinAvg', 'SkinDark', 'meanIAT', 'meanExp',
        'Age', 'height', 'weight', 'position', 'leagueCountry', 'offset'
    ]
    # Some rows may not contain all keep_cols (e.g., missing playerShort); keep what exists and then reindex
    existing = [c for c in keep_cols if c in df.columns]
    df = df[existing].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Fit a negative binomial generalized linear model for count data with an exposure offset.
    # The offset uses the log of the number of games so the model estimates rate (redCards per game).
    # We include SkinDark as the main predictor and control for referee-country bias measures
    # and player/league covariates. C(...) treats categorical variables correctly.

    formula = (
        'redCards ~ SkinDark + meanIAT + meanExp + Age + height + weight '
        '+ C(position) + C(leagueCountry)'
    )

    # Fit Negative Binomial GLM with offset = log(games)
    glm_mod = sm.GLM.from_formula(
        formula,
        data=df,
        family=sm.families.NegativeBinomial(),
        offset=df['offset']
    )
    res = glm_mod.fit()

    # Compute clustered robust covariance matrix by referee (refNum)
    # Using statsmodels' sandwich covariance utilities to obtain cluster-robust cov.
    try:
        clustered_cov = sm.stats.sandwich_covariance.cov_cluster(res, df['refNum'])
    except Exception:
        # If cluster covariance computation fails, fall back to the model's default covariance
        clustered_cov = res.cov_params()

    # Compute clustered standard errors, z-stats, p-values, and confidence intervals
    bse_cluster = np.sqrt(np.diag(clustered_cov))
    params = res.params.copy()
    # Ensure alignment in case of ordering differences
    param_index = params.index
    # If clustered_cov is a ndarray, its order should match params order from res.params
    z_scores = params.values / bse_cluster
    pvalues_cluster = 2 * stats.norm.sf(np.abs(z_scores))
    ci_lower = params.values - 1.96 * bse_cluster
    ci_upper = params.values + 1.96 * bse_cluster

    # Create a summary DataFrame for clustered results
    summary_df = pd.DataFrame({
        'coef': params.values,
        'bse_cluster': bse_cluster,
        'z_cluster': z_scores,
        'pvalue_cluster': pvalues_cluster,
        'ci_lower_95': ci_lower,
        'ci_upper_95': ci_upper
    }, index=param_index)

    # Attach clustered results to the original results object for downstream inspection
    res.cov_cluster = clustered_cov
    res.bse_cluster = pd.Series(bse_cluster, index=param_index)
    res.pvalues_cluster = pd.Series(pvalues_cluster, index=param_index)
    res.summary_cluster_df = summary_df

    # Print a concise clustered-summary and return the results object with clustered covariance.
    try:
        # Print the standard model summary first
        print(res.summary())
    except Exception:
        pass

    try:
        print("\nClustered (by refNum) coefficient summary:")
        print(summary_df)
    except Exception:
        pass

    return res