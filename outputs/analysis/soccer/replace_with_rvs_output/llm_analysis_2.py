from typing import Any
import numpy as np
import pandas as pd
import sklearn
import scipy
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle


df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/soccer/replace_with_rvs_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy
    df = df.copy()

    # Keep only rows with essential fields present
    df = df.dropna(subset=['rater1', 'rater2', 'redCards', 'games'])

    # Compute mean skin tone from two raters (continuous 0-1)
    df['SkinToneMean'] = df[['rater1', 'rater2']].mean(axis=1)

    # Create tercile-based groups to contrast light vs dark; drop the middle tercile to sharpen comparison
    q_low, q_high = df['SkinToneMean'].quantile([1/3, 2/3]).values
    def tone_group(x):
        if x <= q_low:
            return 'Light'
        elif x >= q_high:
            return 'Dark'
        else:
            return 'Middle'
    df['SkinToneGroup'] = df['SkinToneMean'].apply(tone_group)

    # Filter to only Light and Dark groups (exclude middle tercile for clearer comparison)
    df = df[df['SkinToneGroup'].isin(['Light', 'Dark'])].copy()

    # Binary indicator for dark skin (1 = Dark, 0 = Light)
    df['IsDark'] = (df['SkinToneGroup'] == 'Dark').astype(int)
    df['IsLight'] = (df['SkinToneGroup'] == 'Light').astype(int)

    # Parse birthday and compute approximate age in years at season midpoint (2013-01-01)
    # Birthday format in data: 'dd.mm.yyyy'
    df['birthday_parsed'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y', errors='coerce')
    ref_date = pd.to_datetime('2013-01-01')
    df['AgeYears'] = (ref_date - df['birthday_parsed']).dt.days / 365.25

    # Create player-level aggression indicator (yellow cards per game) and ensure games > 0
    df = df[df['games'] > 0]
    df['PlayerAggressionRate'] = df['yellowCards'] / df['games']

    # Keep columns needed for modeling (do not remove others, but ensure these exist)
    required_cols = [
        'playerShort', 'refNum', 'refCountry', 'redCards', 'games',
        'SkinToneMean', 'SkinToneGroup', 'IsDark', 'IsLight',
        'AgeYears', 'height', 'weight', 'goals', 'yellowCards', 'PlayerAggressionRate',
        'position', 'leagueCountry', 'meanIAT', 'meanExp'
    ]

    # Add any missing required columns as NA to avoid KeyErrors downstream
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Final dataframe returned for modeling
    return df.reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Copy to avoid modifying original
    df = df.copy()

    # Select variables used in the model and drop missing values for them
    model_cols = [
        'redCards', 'games', 'IsDark', 'AgeYears', 'height', 'weight',
        'goals', 'yellowCards', 'PlayerAggressionRate', 'position', 'leagueCountry',
        'meanIAT', 'meanExp', 'refNum'
    ]
    df_model = df[model_cols].dropna()

    # Build the formula. Use categorical encoding for position and leagueCountry using C(...)
    formula = (
        'redCards ~ IsDark + AgeYears + height + weight + goals + yellowCards + '
        'PlayerAggressionRate + C(position) + C(leagueCountry) + meanIAT + meanExp'
    )

    # Fit a Negative Binomial GLM with log(games) as an offset (exposure).
    offset = np.log(df_model['games'].astype(float))

    # Fit the model
    model_glm = sm.GLM.from_formula(formula,
                                    data=df_model,
                                    family=sm.families.NegativeBinomial(),
                                    offset=offset)
    fit = model_glm.fit()

    # Obtain cluster-robust standard errors clustered at the referee level (refNum).
    # Some statsmodels versions do not implement get_robustcov_results on GLMResults;
    # compute robust covariance manually and wrap results.
    try:
        # Try cluster-robust sandwich estimator
        cov = sm.stats.sandwich_covariance.cov_cluster(fit, df_model['refNum'])
    except Exception:
        # Fall back to HC1 if clustering fails
        try:
            cov = sm.stats.sandwich_covariance.cov_hc1(fit)
        except Exception:
            # As a last resort, use the model's default covariance
            cov = fit.cov_params()

    bse = np.sqrt(np.diag(cov))
    params = fit.params
    # Guard against mismatched ordering/shapes
    try:
        bse = pd.Series(bse, index=params.index)
    except Exception:
        # If something unexpected happens, coerce to array aligned by position
        bse = pd.Series(bse, index=params.index[:len(bse)])

    z_vals = params / bse
    pvals = 2 * stats.norm.sf(np.abs(z_vals))

    class RobustResults:
        def __init__(self, orig_result, cov_matrix, params, bse, z_vals, pvals):
            self._orig = orig_result
            self.params = params
            self.bse = bse
            self.zvalues = z_vals
            self.pvalues = pvals
            self._cov = cov_matrix
            # Provide commonly accessed attributes
            self.model = getattr(orig_result, 'model', None)

        def cov_params(self):
            return self._cov

        # Delegate summary to original if available
        def summary(self, *args, **kwargs):
            if hasattr(self._orig, 'summary'):
                return self._orig.summary(*args, **kwargs)
            return None

        # Allow access to the original fit if needed
        def original_result(self):
            return self._orig

    robust_res = RobustResults(fit, cov, params, bse, z_vals, pvals)
    return robust_res