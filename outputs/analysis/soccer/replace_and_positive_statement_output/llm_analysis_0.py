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
    Transform the raw dyad dataset into an analysis-ready dataframe.

    Produces the following columns used in modeling:
      - redCards (count outcome)
      - games (exposure)
      - SkinAvg (continuous average of rater1 and rater2)
      - SkinDark (binary: 1 = dark tercile, 0 = light tercile; middle tercile is dropped)
      - age (in years, computed from birthday using reference date 2013-01-01)
      - height, weight, goals, yellowCards, yellowReds, position, leagueCountry, meanIAT, meanExp, refNum
      - hasPhoto: indicator that photo/rating exists

    Notes: we construct a clear contrast between 'dark' and 'light' by taking the top and bottom terciles of the averaged rater score and excluding middle tercile to reduce measurement ambiguity.
    """
    import numpy as np
    import pandas as pd

    df = df.copy()

    # Standard validations and required columns
    required = ['redCards', 'games', 'rater1', 'rater2', 'birthday', 'height', 'weight',
                'goals', 'yellowCards', 'yellowReds', 'position', 'leagueCountry',
                'meanIAT', 'meanExp', 'refNum']
    # Keep only rows that have the primary outcome and exposure
    df = df.loc[df['games'].notna() & (df['games'] > 0) & df['redCards'].notna()].copy()

    # Indicate whether a photo/rating exists
    df['hasPhoto'] = (~df['rater1'].isna()) | (~df['rater2'].isna())

    # Compute average rater score (use available raters; if both missing, result is NaN)
    df['SkinAvg'] = df[['rater1', 'rater2']].mean(axis=1)

    # Parse birthday into datetime and compute age at reference date (season midpoint ~ 2013-01-01)
    # birthday format in schema is 'dd.mm.yyyy'
    df['birthday_dt'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y', errors='coerce')
    ref_date = pd.to_datetime('2013-01-01')
    df['age'] = ((ref_date - df['birthday_dt']).dt.days / 365.25).astype(float)

    # Keep only rows with a skin rating (we only analyze players with photos/ratings) and essential covariates
    df = df.loc[df['SkinAvg'].notna()].copy()

    # Create tercile thresholds to form a clear light vs dark contrast and drop middle tercile
    q33 = df['SkinAvg'].quantile(0.33)
    q67 = df['SkinAvg'].quantile(0.67)

    def skin_label(x):
        if x <= q33:
            return 'light'
        elif x >= q67:
            return 'dark'
        else:
            return 'mid'

    df['SkinTercile'] = df['SkinAvg'].apply(skin_label)

    # Keep only light and dark terciles for clear comparison (drop mid)
    df = df[df['SkinTercile'].isin(['light', 'dark'])].copy()

    # Binary contrast: 1 = dark, 0 = light
    df['SkinDark'] = (df['SkinTercile'] == 'dark').astype(int)

    # Keep only relevant columns for modeling (but don't drop missing covariates yet; model will handle/missing rows dropped by statsmodel)
    keep_cols = ['redCards', 'games', 'SkinAvg', 'SkinDark', 'age', 'height', 'weight',
                 'goals', 'yellowCards', 'yellowReds', 'position', 'leagueCountry',
                 'meanIAT', 'meanExp', 'refNum', 'hasPhoto']

    # If any of these columns are missing in the provided df, create them as NaN so downstream code fails clearly
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan

    out = df.loc[:, keep_cols].copy()

    # Quick type coercions
    out['redCards'] = out['redCards'].astype(int)
    out['games'] = out['games'].astype(float)
    out['SkinAvg'] = out['SkinAvg'].astype(float)
    out['SkinDark'] = out['SkinDark'].astype(int)
    out['age'] = out['age'].astype(float)

    # Return the cleaned, transformed dataframe used in the model
    return out


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a negative binomial regression of redCards on SkinDark (dark vs light) with exposure offset log(games).

    Controls: age, height, weight, goals, yellowCards, meanIAT, meanExp, categorical position, categorical leagueCountry,
    and referee fixed effects (C(refNum)). We cluster-robust the standard errors by refNum.

    Returns a fitted results object with clustered robust covariance.
    """
    import numpy as np
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = df.copy()

    # Drop rows with missing values in variables used by the formula (statsmodels will error otherwise)
    formula_vars = ['redCards', 'games', 'SkinDark', 'age', 'height', 'weight',
                    'goals', 'yellowCards', 'meanIAT', 'meanExp', 'position', 'leagueCountry', 'refNum']
    df_model = df.dropna(subset=formula_vars).copy()

    # Build formula: include position and leagueCountry as categorical, and referee fixed effects C(refNum)
    # Note: C(refNum) will add referee fixed effects to control for referee-specific mean differences.
    formula = (
        'redCards ~ SkinDark + age + height + weight + goals + yellowCards + '
        'meanIAT + meanExp + C(position) + C(leagueCountry) + C(refNum)'
    )

    # Offset is log(games) because redCards is a count over games (exposure)
    df_model['offset'] = np.log(df_model['games'])

    # Fit a GLM with Negative Binomial family
    try:
        glm_nb = smf.glm(formula=formula, data=df_model,
                         family=sm.families.NegativeBinomial(),
                         offset=df_model['offset'])
        res = glm_nb.fit()
    except Exception as e:
        # If NegativeBinomial fails to converge or model matrix is too large due to many ref fixed effects,
        # fall back to Poisson with robust dispersion (Poisson with cluster-robust SE is common for count data as well).
        glm_p = smf.glm(formula=formula, data=df_model,
                        family=sm.families.Poisson(),
                        offset=df_model['offset'])
        res = glm_p.fit()

    # Obtain cluster-robust standard errors clustered by referee (refNum)
    try:
        res_clust = res.get_robustcov_results(cov_type='cluster', groups=df_model['refNum'])
    except Exception:
        # If clustering fails for some reason, return the original result
        res_clust = res

    # Print a concise summary (coef, std err, z, p) and return the clustered-results object
    print(res_clust.summary())
    return res_clust


