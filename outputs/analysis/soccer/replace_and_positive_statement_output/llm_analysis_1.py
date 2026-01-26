from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

# Attempt to read the dataset (path retained from original context)
# If running in a different environment, callers can ignore this or replace path.
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/replace_and_positive_statement_output/soccer.csv')
except Exception:
    df = pd.DataFrame()  # empty placeholder if file not found


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dyad dataframe into the analysis-ready dataframe.

    Steps:
    - Parse dates and compute player age
    - Compute mean skin rating from two raters (skin_avg)
    - Keep only players with skin ratings and non-missing games
    - Define extreme-sample: bottom quartile (light) vs top quartile (dark)
    - Create DarkSkin binary (1 = dark quartile, 0 = light quartile)
    - Create per-game rates for goals and yellow cards
    - Ensure categorical fields are tidy
    - Add log(games) column for model offset
    """
    df = df.copy()

    # Parse birthday; tolerate malformed entries
    if 'birthday' in df.columns:
        df['birthday_parsed'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y', errors='coerce')
    else:
        df['birthday_parsed'] = pd.NaT

    # Season is 2012-2013 -> pick mid-season reference date (2012-09-01)
    ref_date = pd.to_datetime('2012-09-01')
    df['age'] = ((ref_date - df['birthday_parsed']).dt.days / 365.25).astype(float)

    # Compute mean skin rating from two raters (both are normalized 0..1). If one rater missing, mean will use the available rater.
    if 'rater1' in df.columns and 'rater2' in df.columns:
        df['skin_avg'] = df[['rater1', 'rater2']].mean(axis=1)
    elif 'rater1' in df.columns:
        df['skin_avg'] = df['rater1'].astype(float)
    elif 'rater2' in df.columns:
        df['skin_avg'] = df['rater2'].astype(float)
    else:
        df['skin_avg'] = np.nan

    # Drop rows with missing essential values: skin ratings, redCards, games
    required_initial = [c for c in ['skin_avg', 'redCards', 'games'] if c in df.columns]
    if required_initial:
        df = df.dropna(subset=required_initial)

    # Ensure games > 0 (schema indicates min 1, but be defensive)
    if 'games' in df.columns:
        df = df[df['games'] > 0]

    # Compute per-game rates for goals and yellow cards (controls). Defensive division by games handled since games>0
    if 'goals' in df.columns and 'games' in df.columns:
        df['goals_rate'] = df['goals'] / df['games']
    else:
        df['goals_rate'] = np.nan

    if 'yellowCards' in df.columns and 'games' in df.columns:
        df['yellowCards_rate'] = df['yellowCards'] / df['games']
    else:
        df['yellowCards_rate'] = np.nan

    # Determine quartiles of skin_avg and keep only extremes (top and bottom quartile) to contrast dark vs light players
    if 'skin_avg' in df.columns and not df['skin_avg'].isna().all():
        q25 = df['skin_avg'].quantile(0.25)
        q75 = df['skin_avg'].quantile(0.75)
        df = df[(df['skin_avg'] <= q25) | (df['skin_avg'] >= q75)].copy()
    else:
        # If skin_avg missing entirely, return empty dataframe (can't form DarkSkin)
        return pd.DataFrame(columns=[
            'DarkSkin', 'redCards', 'games', 'skin_avg', 'age', 'height', 'weight',
            'position', 'goals_rate', 'yellowCards_rate', 'meanIAT', 'meanExp',
            'leagueCountry', 'refNum', 'log_games'
        ])

    # Binary indicator: 1 = dark (top quartile), 0 = light (bottom quartile)
    df['DarkSkin'] = (df['skin_avg'] >= q75).astype(int)

    # Clean position and leagueCountry to avoid missing category issues
    if 'position' in df.columns:
        df['position'] = df['position'].fillna('Unknown')
    else:
        df['position'] = 'Unknown'

    if 'leagueCountry' in df.columns:
        df['leagueCountry'] = df['leagueCountry'].fillna('Unknown')
    else:
        df['leagueCountry'] = 'Unknown'

    # Keep covariates used in modelling; preserve refNum for clustered SE
    # Add offset column: log(games)
    if 'games' in df.columns:
        # guard against zero or negative games already filtered above
        df['log_games'] = np.log(df['games'])
    else:
        df['log_games'] = np.nan

    # Final columns used in modeling (kept for clarity)
    model_cols = ['playerShort', 'player', 'club', 'leagueCountry', 'birthday_parsed', 'age', 'height', 'weight',
                  'position', 'games', 'log_games', 'victories', 'ties', 'defeats', 'goals', 'goals_rate',
                  'yellowCards', 'yellowCards_rate', 'yellowReds', 'redCards', 'photoID', 'rater1', 'rater2',
                  'skin_avg', 'DarkSkin', 'refNum', 'refCountry', 'meanIAT', 'nIAT', 'seIAT', 'meanExp', 'nExp', 'seExp']

    # Some columns may not exist in certain input variants; only keep those that exist
    model_cols = [c for c in model_cols if c in df.columns]
    df = df[model_cols]

    # Final defensive drop of any remaining NAs in key modeling columns
    required_for_model = ['redCards', 'games', 'DarkSkin', 'age', 'height', 'weight', 'position', 'leagueCountry', 'meanIAT', 'meanExp', 'refNum']
    # Only require those that exist in df (we must not change required column names, but avoid KeyError)
    required_for_model = [c for c in required_for_model if c in df.columns]
    if required_for_model:
        df = df.dropna(subset=required_for_model)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative binomial regression predicting red card counts in a player-referee dyad.

    Approach:
    - Outcome: redCards (count)
    - Exposure: games (used as log offset)
    - Key predictor: DarkSkin (1 = top quartile dark, 0 = bottom quartile light)
    - Controls: age, height, weight, goals_rate, yellowCards_rate, categorical position, leagueCountry,
      country-level bias measures meanIAT and meanExp.
    - Inference: cluster-robust standard errors by refNum (referee ID) to account for non-independence of dyads involving the same referee.

    Returns a dictionary containing the original fitted results object and robust inference summaries.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_cluster, cov_hc1
    from scipy import stats as _stats

    df = df.copy()

    # Ensure categorical variables are treated as such
    if 'position' in df.columns:
        df['position'] = df['position'].astype('category')
    else:
        df['position'] = 'Unknown'
        df['position'] = df['position'].astype('category')

    if 'leagueCountry' in df.columns:
        df['leagueCountry'] = df['leagueCountry'].astype('category')
    else:
        df['leagueCountry'] = 'Unknown'
        df['leagueCountry'] = df['leagueCountry'].astype('category')

    # Build formula. We use C() for categorical variables (position & leagueCountry).
    # Keep the conceptual variable names exactly as specified.
    formula = 'redCards ~ DarkSkin + age + height + weight + goals_rate + yellowCards_rate + C(position) + meanIAT + meanExp + C(leagueCountry)'

    # Fit Negative Binomial GLM with an offset = log(games)
    # statsmodels' GLM accepts offset as a numpy array
    offset = df['log_games'] if 'log_games' in df.columns else np.log(df['games'])

    model_glm = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=offset)
    res = model_glm.fit()

    # Attempt to compute cluster-robust covariance clustered on refNum.
    # Some statsmodels installations may not provide get_robustcov_results on GLMResults;
    # compute the sandwich covariance explicitly instead.
    cov_matrix = None
    cov_type_used = None
    if 'refNum' in df.columns:
        try:
            cov_matrix = cov_cluster(res, df['refNum'].values)
            cov_type_used = 'cluster'
        except Exception:
            # Fallback to HC1 robust covariance (heteroskedasticity-consistent) if clustering fails
            cov_matrix = cov_hc1(res)
            cov_type_used = 'HC1'
    else:
        # No clustering variable; use HC1 as default robust covariance
        cov_matrix = cov_hc1(res)
        cov_type_used = 'HC1'

    # Derive inference quantities from covariance matrix
    params = res.params
    # Ensure alignment in case of shape issues
    try:
        bse = np.sqrt(np.diag(cov_matrix))
    except Exception:
        # Fallback to model-provided standard errors if covariance matrix computation failed
        bse = res.bse.values if hasattr(res.bse, 'values') else np.asarray(res.bse)

    # Avoid division by zero in edge cases
    with np.errstate(divide='ignore', invalid='ignore'):
        z_stats = params / bse
    pvalues = 2 * (1 - _stats.norm.cdf(np.abs(z_stats)))
    crit = _stats.norm.ppf(0.975)
    ci_lower = params - crit * bse
    ci_upper = params + crit * bse

    # Organize a tidy summary table for printing
    summary_table = pd.DataFrame({
        'coef': params,
        'std_err': bse,
        'z': z_stats,
        'P>|z|': pvalues,
        'CI_2.5': ci_lower,
        'CI_97.5': ci_upper
    })

    print(f"Robust covariance type used: {cov_type_used}")
    print(summary_table)

    # Prepare output dict similar to original expectations
    summary_dict = {
        'original_result': res,
        'cov_type': cov_type_used,
        'cov_matrix': cov_matrix,
        'coef': params.to_dict(),
        'se': pd.Series(bse, index=params.index).to_dict(),
        'pvalues': pd.Series(pvalues, index=params.index).to_dict(),
        'conf_int': pd.DataFrame({'2.5%': ci_lower, '97.5%': ci_upper}, index=params.index).to_dict()
    }

    return summary_dict