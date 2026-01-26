from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
from types import SimpleNamespace
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hc1

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/positive_leading_statement_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dyad-level dataframe into analysis-ready dataframe containing:
      - SkinAvg: average of rater1 and rater2 (0-1 scale)
      - SkinDark: binary indicator = 1 for 'Dark' (SkinAvg >= 0.75), 0 for 'Light' (SkinAvg <= 0.25)
      - Filter to only 'Dark' and 'Light' extremes (drops middle/ambiguous skin tones)
      - PlayerAge: age in years at season midpoint (2013-01-01 used as reference)
      - yellowRate, goalsRate: per-game rates for behavior controls
      - log_games: natural log of games (offset)
    Returns dataframe with all columns required by the model.
    """
    # make a copy to avoid modifying input in place
    df = df.copy()

    # Ensure required columns exist
    required = ['rater1', 'rater2', 'redCards', 'games', 'birthday', 'yellowCards', 'goals',
                'position', 'leagueCountry', 'height', 'weight', 'meanIAT', 'meanExp', 'refNum', 'playerShort']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in input dataframe: {missing}")

    # Compute average skin rating from two raters (rater values are normalized to 0-1 in dataset description)
    df['SkinAvg'] = df[['rater1', 'rater2']].mean(axis=1)

    # Define extremes: Light <= 0.25, Dark >= 0.75 (these correspond to bottom and top categories on 5-point scale)
    def skin_cat(x):
        if pd.isna(x):
            return pd.NA
        if x <= 0.25:
            return 'Light'
        if x >= 0.75:
            return 'Dark'
        return 'Other'

    df['SkinCat'] = df['SkinAvg'].apply(skin_cat)

    # Filter to only extreme groups (question asks Dark vs Light)
    df = df[df['SkinCat'].isin(['Dark', 'Light'])].copy()

    # Binary indicator for dark skin (1 = Dark, 0 = Light)
    df['SkinDark'] = (df['SkinCat'] == 'Dark').astype(int)

    # Clean numeric columns: ensure games is numeric and >=1 (dataset description min=1)
    df['games'] = pd.to_numeric(df['games'], errors='coerce')
    # Drop any rows with no games
    df = df[df['games'].notna() & (df['games'] > 0)].copy()

    # Ensure redCards is numeric
    df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce').fillna(0).astype(int)

    # Compute per-game rates for behavioral controls (safe division: games >=1)
    df['yellowRate'] = pd.to_numeric(df.get('yellowCards', pd.Series(np.nan, index=df.index)), errors='coerce') / df['games'].astype(float)
    df['goalsRate'] = pd.to_numeric(df.get('goals', pd.Series(np.nan, index=df.index)), errors='coerce') / df['games'].astype(float)

    # Compute log offset for exposure
    df['log_games'] = np.log(df['games'].astype(float))

    # Parse birthday into datetime and compute age at season midpoint (use 2013-01-01 as representative)
    # birthday format in schema: 'dd.mm.yyyy'
    df['birthday_parsed'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y', errors='coerce')
    # If parsing fails for any rows, try generic parsing
    mask_bad_birth = df['birthday_parsed'].isna() & df['birthday'].notna()
    if mask_bad_birth.any():
        df.loc[mask_bad_birth, 'birthday_parsed'] = pd.to_datetime(df.loc[mask_bad_birth, 'birthday'], dayfirst=True, errors='coerce')

    # Age at 2013-01-01 (season midpoint between 2012 and 2013)
    ref_date = pd.to_datetime('2013-01-01')
    df['PlayerAge'] = ((ref_date - df['birthday_parsed']).dt.days / 365.25).round(2)

    # Keep only columns necessary for modeling (but preserve some identifiers)
    keep_cols = ['playerShort', 'SkinAvg', 'SkinCat', 'SkinDark', 'redCards', 'games', 'log_games',
                 'PlayerAge', 'position', 'leagueCountry', 'height', 'weight', 'yellowRate', 'goalsRate',
                 'meanIAT', 'meanExp', 'refNum']
    # Some of these columns may not have existed before (already computed), ensure all present
    for c in keep_cols:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[keep_cols].reset_index(drop=True)

    # Drop rows with missing values in any predictor or outcome that would break model fitting
    model_required = ['SkinDark', 'redCards', 'games', 'log_games', 'PlayerAge', 'position', 'leagueCountry', 'refNum']
    df = df.dropna(subset=model_required).copy()

    # Ensure numeric types
    df['SkinDark'] = df['SkinDark'].astype(int)
    df['redCards'] = df['redCards'].astype(int)
    df['games'] = df['games'].astype(float)
    df['log_games'] = df['log_games'].astype(float)
    df['PlayerAge'] = pd.to_numeric(df['PlayerAge'], errors='coerce')
    df['height'] = pd.to_numeric(df['height'], errors='coerce')
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df['yellowRate'] = pd.to_numeric(df['yellowRate'], errors='coerce')
    df['goalsRate'] = pd.to_numeric(df['goalsRate'], errors='coerce')
    df['meanIAT'] = pd.to_numeric(df['meanIAT'], errors='coerce')
    df['meanExp'] = pd.to_numeric(df['meanExp'], errors='coerce')
    df['refNum'] = pd.to_numeric(df['refNum'], errors='coerce').astype(int)

    # Drop final rows with missing numeric covariates (the model will need them)
    df = df.dropna(subset=['PlayerAge', 'position', 'leagueCountry', 'meanIAT', 'meanExp'])

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a generalized linear model for red card counts per dyad with exposure offset = log(games).
    Steps:
      1. Fit Poisson GLM with formula including SkinDark and controls.
      2. Compute dispersion (Pearson chi2 / df_resid). If dispersion > 1.5, refit with Negative Binomial.
      3. Compute cluster-robust standard errors clustered by referee (refNum).
      4. Return both the fitted model and the clustered-results object plus diagnostics (dispersion, IRR for SkinDark).

    The function expects the dataframe produced by transform(), containing the columns named in the conceptual variables.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hc1

    # Formula: redCards predicted by SkinDark plus controls and categorical fixed effects for position and leagueCountry
    formula = (
        'redCards ~ SkinDark + PlayerAge + height + weight + yellowRate + goalsRate + meanIAT + meanExp '
        '+ C(position) + C(leagueCountry)'
    )

    # Fit Poisson GLM with offset (exposure = games -> offset = log(games))
    poisson_model = smf.glm(formula=formula, data=df, family=sm.families.Poisson(), offset=df['log_games']).fit()

    # Compute dispersion: Pearson chi2 / df_resid
    # resid_pearson is available on GLMResults
    pearson_chi2 = np.sum(poisson_model.resid_pearson ** 2)
    dispersion = pearson_chi2 / poisson_model.df_resid if poisson_model.df_resid > 0 else np.nan

    final_model = poisson_model
    model_family = 'Poisson'

    # If evidence of overdispersion, fit Negative Binomial instead
    if (not np.isnan(dispersion)) and (dispersion > 1.5):
        try:
            nb_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=df['log_games']).fit()
            final_model = nb_model
            model_family = 'NegativeBinomial'
        except Exception as e:
            # If NB fails, keep Poisson but note it
            print('NegativeBinomial fit failed; keeping Poisson. Error:', e)

    # Compute cluster-robust standard errors clustered by referee (refNum)
    # Some statsmodels versions do not provide get_robustcov_results on GLMResults,
    # so compute clustered covariance manually and expose params and bse.
    try:
        # cov_cluster expects (results, groups)
        cov = cov_cluster(final_model, df['refNum'])
    except Exception:
        # fallback to HC1 robust covariance if clustering fails
        try:
            cov = cov_hc1(final_model)
        except Exception:
            # final fallback: use model's covariance matrix
            cov = final_model.cov_params()

    params = final_model.params
    bse = pd.Series(np.sqrt(np.diag(cov)), index=params.index)

    clustered = SimpleNamespace(params=params, bse=bse, cov=cov)

    # Compute incidence rate ratios (IRR) and CIs for key parameter SkinDark using clustered cov results
    params = clustered.params
    bse = clustered.bse
    if 'SkinDark' in params.index:
        coef = params['SkinDark']
        se = bse['SkinDark']
        irr = np.exp(coef)
        # 95% CI on log scale then transform
        ci_lower = np.exp(coef - 1.96 * se)
        ci_upper = np.exp(coef + 1.96 * se)
    else:
        irr = np.nan
        ci_lower = np.nan
        ci_upper = np.nan

    # Pack results in a dictionary for easy downstream inspection
    results = {
        'final_model': final_model,                  # statsmodels GLMResults (Poisson or NB)
        'clustered_results': clustered,              # object with params and bse based on clustered cov
        'model_family': model_family,
        'dispersion': dispersion,
        'SkinDark_IRR': irr,
        'SkinDark_IRR_CI': (ci_lower, ci_upper),
        'formula': formula
    }

    return results