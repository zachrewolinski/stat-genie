from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/replace_and_positive_statement_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe. Returns a dataframe containing the columns used by the model.

    Key steps:
    - Parse birthday and compute age (reference date = 2013-01-01, season midpoint)
    - Compute mean skin rating from rater1 and rater2
    - Create a binary SkinDark (1 = dark, 0 = light) by thresholding mean rating (<=0.4 = light; >=0.6 = dark); drop middle/category-ambiguous cases
    - Keep only dyads with non-missing skin ratings and with games > 0 (we need exposure > 0)
    - Ensure numeric types for redCards and games
    - Return a reduced dataframe with all columns used in modeling
    """
    df = df.copy()

    # Parse birthday (format dd.mm.yyyy per schema) and compute age at season midpoint
    df['birthday'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y', errors='coerce')
    reference_date = pd.to_datetime('2013-01-01')  # midpoint of 2012-2013 season
    df['age'] = (reference_date - df['birthday']).dt.days / 365.25

    # Compute mean skin rating from two independent raters
    # rater1 and rater2 are normalized 5-point ratings (0.0 - 1.0) per schema
    if 'rater1' in df.columns and 'rater2' in df.columns:
        df['skin_mean'] = df[['rater1', 'rater2']].mean(axis=1)
    else:
        # If one of the raters is missing, compute mean of available; if both missing, will be NaN
        df['skin_mean'] = pd.NA
        if 'rater1' in df.columns:
            df['skin_mean'] = df['rater1']
        if 'rater2' in df.columns:
            df['skin_mean'] = df['skin_mean'].combine(df['rater2'], lambda a, b: np.nan if pd.isna(a) and pd.isna(b) else np.nanmean([a, b]))

    # Filter out rows with missing skin ratings or missing game counts
    df = df[df['skin_mean'].notnull()]
    df = df[df['games'].notnull()]

    # Ensure games is numeric and positive; if not, coerce then drop non-positive
    df['games'] = pd.to_numeric(df['games'], errors='coerce')
    df = df[df['games'] > 0]

    # Create a clear binary contrast: light (0) vs dark (1). Exclude middle/ambiguous cases to make groups distinct.
    # Thresholds chosen so that ratings near the extremes are classified: <=0.4 -> light; >=0.6 -> dark; otherwise -> NaN
    df['SkinDark'] = df['skin_mean'].apply(lambda x: 1 if x >= 0.6 else (0 if x <= 0.4 else np.nan))
    df = df[df['SkinDark'].notnull()]
    df['SkinDark'] = df['SkinDark'].astype(int)

    # Ensure redCards numeric (count) and non-negative
    if 'redCards' in df.columns:
        df['redCards'] = pd.to_numeric(df['redCards'], errors='coerce').fillna(0).astype(int)
    else:
        # If redCards missing, create zeros (will lead to trivial model but keeps shape)
        df['redCards'] = 0

    # Ensure numeric controls if present
    if 'height' in df.columns:
        df['height'] = pd.to_numeric(df['height'], errors='coerce')
    if 'weight' in df.columns:
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    if 'meanIAT' in df.columns:
        df['meanIAT'] = pd.to_numeric(df['meanIAT'], errors='coerce')
    if 'meanExp' in df.columns:
        df['meanExp'] = pd.to_numeric(df['meanExp'], errors='coerce')

    # Keep only the columns needed for the model (and some identifiers for diagnostics)
    keep_cols = [
        'playerShort', 'player', 'club', 'leagueCountry', 'birthday', 'age', 'height', 'weight',
        'position', 'games', 'redCards', 'photoID', 'rater1', 'rater2', 'skin_mean', 'SkinDark',
        'refNum', 'refCountry', 'meanIAT', 'meanExp'
    ]

    # Some rows may be missing many control variables; we keep them but the model will handle missingness implicitly (statsmodels will drop rows with missing data in the formula). If desired, further imputation can be added.
    df = df.loc[:, [c for c in keep_cols if c in df.columns]]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a negative-binomial GLM for red card counts with games as exposure (offset).

    Model specification:
    - Outcome: redCards (count)
    - Exposure/offset: log(games)
    - Key predictor: SkinDark (1 = dark, 0 = light)
    - Controls: position (categorical), age, height, weight, meanIAT, meanExp, leagueCountry (categorical)
    - Cluster-robust standard errors clustered by referee id (refNum)

    Returns a dict with the fitted (cluster-robust when possible) results object and a small table of incidence rate ratios (IRRs)
    and 95% CIs to aid interpretation.
    """
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    # Work on a copy
    data = df.copy()

    # Ensure required columns exist
    required_cols = ['redCards', 'games', 'SkinDark', 'position', 'age', 'height', 'weight', 'meanIAT', 'meanExp', 'leagueCountry', 'refNum']
    # We won't enforce all required columns here, but the formula below expects certain ones to be present in the dataframe.
    # Drop rows with missing essential variables for modeling: outcome, exposure, key predictor
    data = data.dropna(subset=['redCards', 'games', 'SkinDark'])

    # Build formula. Use categorical encoding for position and leagueCountry via C(...)
    formula = 'redCards ~ SkinDark + C(position) + age + height + weight + meanIAT + meanExp + C(leagueCountry)'

    # Offset: log of games (exposure)
    offset = np.log(data['games'])

    # Fit Negative Binomial GLM.
    # To obtain cluster-robust standard errors compatibly across statsmodels versions, request robust cov directly in fit when possible.
    try:
        if 'refNum' in data.columns:
            # Attempt to fit with clustered covariance (this asks fit to compute cov_params with clustering)
            nb_res = sm.GLM.from_formula(formula, data=data, family=sm.families.NegativeBinomial(), offset=offset).fit(cov_type='cluster', cov_kwds={'groups': data['refNum']})
        else:
            nb_res = sm.GLM.from_formula(formula, data=data, family=sm.families.NegativeBinomial(), offset=offset).fit()
    except Exception:
        # Fallback: fit without cluster argument, then request HC1 robust cov via fit option if possible
        try:
            nb_res = sm.GLM.from_formula(formula, data=data, family=sm.families.NegativeBinomial(), offset=offset).fit(cov_type='HC1')
        except Exception:
            # Final fallback: plain fit
            nb_res = sm.GLM.from_formula(formula, data=data, family=sm.families.NegativeBinomial(), offset=offset).fit()

    # nb_res now should be a fitted results object whose covariance reflects clustering or robust method if supported by the environment.
    res_clust = nb_res

    # Create an IRR (incidence rate ratio) table for easy interpretation
    params = res_clust.params
    conf = res_clust.conf_int()
    irr = np.exp(params)
    # conf is a DataFrame-like with two columns [0,1], handle accordingly
    try:
        irr_ci_lower = np.exp(conf.iloc[:, 0])
        irr_ci_upper = np.exp(conf.iloc[:, 1])
    except Exception:
        # If conf is not indexable as above, attempt direct exponent
        irr_ci_lower = np.exp(conf[0])
        irr_ci_upper = np.exp(conf[1])

    irr_table = pd.DataFrame({
        'IRR': irr,
        'IRR_CI_lower': irr_ci_lower,
        'IRR_CI_upper': irr_ci_upper,
        'pvalue': res_clust.pvalues
    })

    # Return the (possibly robust) results object and the IRR table
    return {
        'results_object': res_clust,
        'irr_table': irr_table
    }