from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/replace_and_positive_statement_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis-ready dataframe.

    Outputs a dataframe that contains at least the following columns used in modeling:
      - redCards (dependent variable, integer count)
      - games (exposure)
      - SkinToneScore (continuous mean of rater1 and rater2)
      - SkinToneBin (categorical: 'Dark' or 'Light')
      - age, height, weight
      - position_simp (simplified position)
      - leagueCountry, meanIAT, meanExp, refNum

    Filtering decisions:
      - Keep only dyads with a player photo rated by both raters (rater1 and rater2 non-missing)
      - Keep only dyads with games > 0
      - Focus the primary analysis on players in the extreme skin-tone groups (top 20% vs bottom 20%) to increase contrast between 'Dark' and 'Light'.
    """
    df = df.copy()

    # Basic required columns present check (will raise if missing)
    required = ['redCards', 'games', 'rater1', 'rater2', 'birthday', 'height', 'weight', 'position', 'leagueCountry', 'meanIAT', 'meanExp', 'refNum']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for transform: {missing_cols}")

    # Keep only dyads with at least one game (exposure must be > 0) and non-missing redCards
    df = df[df['games'].notnull()]
    df = df[df['games'] > 0]
    df = df[df['redCards'].notnull()]

    # Require both rater scores and a photo to compute skin tone
    # photoID may be missing for some players; require both raters instead
    df = df[df['rater1'].notnull() & df['rater2'].notnull()]

    # Compute continuous skin tone score as the mean of the two raters
    df['SkinToneScore'] = df[['rater1', 'rater2']].mean(axis=1)

    # Create extreme-group SkinToneBin: top 20% -> 'Dark', bottom 20% -> 'Light', else 'Medium'
    q_low = df['SkinToneScore'].quantile(0.20)
    q_high = df['SkinToneScore'].quantile(0.80)

    def tone_bin(x, low=q_low, high=q_high):
        if pd.isnull(x):
            return pd.NA
        if x <= low:
            return 'Light'
        if x >= high:
            return 'Dark'
        return 'Medium'

    df['SkinToneBin'] = df['SkinToneScore'].apply(tone_bin)

    # Keep only extreme groups for the primary test (contrast Dark vs Light)
    df = df[df['SkinToneBin'].isin(['Dark', 'Light'])].copy()

    # Parse birthday to compute age at a reference date (use 2013-01-01 as season midpoint)
    ref_date = pd.to_datetime('2013-01-01')
    # Try multiple possible formats
    if not np.issubdtype(df['birthday'].dtype, np.datetime64):
        # Some dates are in dd.mm.yyyy format per schema
        df['birthday'] = pd.to_datetime(df['birthday'], dayfirst=True, errors='coerce')
    df['age'] = (ref_date - df['birthday']).dt.days / 365.25

    # Keep numeric height and weight as-is, but coerce to numeric if necessary
    df['height'] = pd.to_numeric(df['height'], errors='coerce')
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

    # Simplify position into broad categories
    def simplify_position(pos):
        if pd.isnull(pos):
            return 'Other'
        s = str(pos).lower()
        if 'forward' in s or 'striker' in s or 'attacker' in s or 'winger' in s:
            return 'Forward'
        if 'mid' in s:
            return 'Midfielder'
        if 'def' in s or 'back' in s:
            return 'Defender'
        if 'goal' in s or 'keeper' in s:
            return 'Goalkeeper'
        return 'Other'

    df['position_simp'] = df['position'].apply(simplify_position)

    # leagueCountry kept as-is (categorical)
    df['leagueCountry'] = df['leagueCountry'].astype('category')

    # Ensure meanIAT and meanExp are numeric
    df['meanIAT'] = pd.to_numeric(df['meanIAT'], errors='coerce')
    df['meanExp'] = pd.to_numeric(df['meanExp'], errors='coerce')

    # Keep only rows with non-missing essential covariates for the model
    essential = ['redCards', 'games', 'SkinToneScore', 'SkinToneBin', 'age', 'height', 'weight', 'position_simp', 'leagueCountry', 'meanIAT', 'meanExp', 'refNum']
    df = df.dropna(subset=essential)

    # Reset index and return abbreviated dataframe (but keep any other columns still present)
    df = df.reset_index(drop=True)

    # Make sure redCards and games are integer-like
    df['redCards'] = df['redCards'].astype(int)
    df['games'] = df['games'].astype(int)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a negative binomial GLM for redCards counts with exposure = games.

    Primary coefficient of interest: SkinToneBin (Dark vs Light). We cluster standard errors by referee (refNum)
    to account for non-independence of dyads involving the same referee.

    Returns the fitted model result object with cluster-robust covariances.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import numpy as np

    # Formula: redCards count predicted by SkinToneBin (Dark vs Light) and controls
    # Use categorical variables for position and leagueCountry
    formula = (
        'redCards ~ C(SkinToneBin, Treatment("Light")) '
        '+ age + height + weight '
        '+ C(position_simp) + C(leagueCountry) '
        '+ meanIAT + meanExp'
    )

    # Offset = log(games) to model rate per game
    offset = np.log(df['games'].astype(float))

    # Fit Negative Binomial GLM
    model_glm = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=offset)
    res = model_glm.fit()

    # Obtain cluster-robust (by refNum) covariance estimates
    # Use get_robustcov_results to attach clustered covariances to the results
    try:
        res_cluster = res.get_robustcov_results(cov_type='cluster', groups=df['refNum'])
    except Exception:
        # Fallback: if clustering fails, return regular results but note this in an attribute
        res_cluster = res
        res_cluster._cluster_warning = 'Cluster-robust covariance estimation failed; returning unclustered results.'

    # Additionally compute incidence rate ratios (IRR) and their clustered CIs for convenience
    params = res_cluster.params
    conf = res_cluster.conf_int()
    irr = np.exp(params)
    irr_ci_lower = np.exp(conf[0])
    irr_ci_upper = np.exp(conf[1])

    # Pack summary information in a small dict for easy inspection
    summary_dict = {
        'model': res_cluster,
        'params': params,
        'conf_int': conf,
        'IRR': irr,
        'IRR_ci_lower': irr_ci_lower,
        'IRR_ci_upper': irr_ci_upper,
        'formula': formula,
        'n_obs': int(res_cluster.nobs)
    }

    # Print concise results for the primary variable
    try:
        coef_name = 'C(SkinToneBin, Treatment("Light"))[T.Dark]'
        coef = params.get(coef_name, None)
        if coef is not None:
            print('Primary test: Dark vs Light')
            print('  coef (log-IRR) =', coef)
            print('  IRR =', float(irr[coef_name]))
            print('  95% CI (IRR) =', float(irr_ci_lower[coef_name]), '-', float(irr_ci_upper[coef_name]))
        else:
            print('Warning: expected coefficient name not found in model parameters. Available params:', list(params.index))
    except Exception as e:
        print('Could not print primary coefficient summary:', e)

    return summary_dict


