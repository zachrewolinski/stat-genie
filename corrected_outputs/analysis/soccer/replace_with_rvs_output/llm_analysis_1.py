from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

# Optional top-level read (preserve original behavior if executed as a script)
# If this path is not present in the environment, users can still import the module
# and call transform/model with their own dataframe.
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/replace_with_rvs_output/soccer.csv')
except Exception:
    df = None

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dyad dataset into variables required for modeling the effect of skin tone on red cards.

    Outputs (added/modified columns):
    - SkinToneMean: mean of rater1 and rater2 (continuous, 0-1, higher = darker)
    - SkinToneBin: categorical label 'Dark' or 'Light' (top/bottom 20% of SkinToneMean)
    - SkinToneBin_Dark: binary indicator (1 if Dark, 0 if Light)
    - Age: age in years at reference date (2012-09-01)
    - log_games: log(games) to use as offset in count model
    - position: ensured to be categorical

    Rows with missing essential values are dropped.
    """
    import numpy as np
    import pandas as pd

    # Make a copy to avoid modifying original
    df = df.copy()

    # Essential columns required for analysis
    required = ['redCards', 'games', 'rater1', 'rater2', 'birthday', 'refNum']
    df = df.dropna(subset=required)

    # Create continuous skin tone measure: mean of rater1 and rater2
    df['SkinToneMean'] = df[['rater1', 'rater2']].mean(axis=1)

    # Remove rows with extreme/missing SkinToneMean
    df = df[~df['SkinToneMean'].isna()]

    # Define dark vs light groups using empirical 20th and 80th percentiles
    q_low = df['SkinToneMean'].quantile(0.20)
    q_high = df['SkinToneMean'].quantile(0.80)

    def bin_skin(x, ql=q_low, qh=q_high):
        if x <= ql:
            return 'Light'
        elif x >= qh:
            return 'Dark'
        else:
            return 'Mid'

    df['SkinToneBin'] = df['SkinToneMean'].apply(bin_skin)

    # Keep only the extreme groups to answer the dark vs light question
    df = df[df['SkinToneBin'].isin(['Dark', 'Light'])].copy()

    # Binary indicator for Dark (1) vs Light (0)
    df['SkinToneBin_Dark'] = (df['SkinToneBin'] == 'Dark').astype(int)

    # Parse birthday and compute age at start of 2012-2013 season (use 2012-09-01)
    # birthday format in schema is dd.mm.yyyy
    df['birthday'] = pd.to_datetime(df['birthday'], format='%d.%m.%Y', errors='coerce')
    ref_date = pd.to_datetime('2012-09-01')
    df['Age'] = (ref_date - df['birthday']).dt.days / 365.25

    # Ensure numeric covariates are numeric
    for col in ['height', 'weight', 'yellowCards', 'goals', 'games', 'redCards', 'meanIAT', 'meanExp']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing core numeric covariates after coercion
    df = df.dropna(subset=['games', 'redCards', 'Age', 'refNum', 'SkinToneMean', 'SkinToneBin_Dark'])

    # Create log(games) offset for modeling. games should be >=1 per schema; guard against zeros.
    df['games'] = df['games'].astype(float)
    df = df[df['games'] > 0]
    df['log_games'] = np.log(df['games'])

    # Ensure position and leagueCountry are categorical
    if 'position' in df.columns:
        df['position'] = df['position'].astype('category')
    if 'leagueCountry' in df.columns:
        df['leagueCountry'] = df['leagueCountry'].astype('category')

    # Final: keep columns needed for modeling and diagnostics
    model_cols = [
        'playerShort', 'player', 'club', 'leagueCountry', 'birthday', 'Age', 'height', 'weight',
        'position', 'games', 'log_games', 'victories', 'ties', 'defeats', 'goals', 'yellowCards',
        'yellowReds', 'redCards', 'photoID', 'SkinToneMean', 'SkinToneBin', 'SkinToneBin_Dark',
        'refNum', 'refCountry', 'meanIAT', 'nIAT', 'seIAT', 'meanExp', 'nExp', 'seExp'
    ]

    # Keep only columns that exist in the dataframe (schema may vary)
    model_cols = [c for c in model_cols if c in df.columns]
    df = df[model_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count regression to test whether darker-skinned players receive more red cards than lighter-skinned players.

    We fit a Negative Binomial GLM with the number of red cards (redCards) as the dependent variable,
    SkinToneBin_Dark (Dark vs Light) as the primary independent variable, and include SkinToneMean
    as a continuous robustness predictor. The number of games in the dyad is included as an offset (log scale).

    Controls: yellowCards, goals, Age, height, weight, categorical position, leagueCountry, meanIAT, meanExp.
    Standard errors are clustered by referee (refNum) to account for non-independence of dyads handled by the same referee.

    Returns the fitted model results object (statsmodels result). Prints a model summary as well.
    """
    import numpy as np
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import pandas as pd

    # Ensure required columns exist
    required_cols = ['redCards', 'log_games', 'SkinToneBin_Dark', 'SkinToneMean', 'refNum']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula: primary predictor is SkinToneBin_Dark; include SkinToneMean as continuous robustness covariate.
    # Add controls if present in df
    controls = []
    for c in ['yellowCards', 'goals', 'Age', 'height', 'weight', 'meanIAT', 'meanExp']:
        if c in df.columns:
            controls.append(c)

    # Categorical controls (will be expanded by patsy): position and leagueCountry
    cat_controls = []
    for c in ['position', 'leagueCountry']:
        if c in df.columns:
            cat_controls.append(f'C({c})')

    rhs_terms = ['SkinToneBin_Dark', 'SkinToneMean'] + controls + cat_controls
    formula = 'redCards ~ ' + ' + '.join(rhs_terms)

    # Prepare the dataframe used for fitting: drop rows with NA in any variable used by the model,
    # and ensure refNum has no missing values. This alignment prevents mismatch between model observations
    # and the groups array used for clustered SEs.
    # Determine which raw columns need to be NA-free for the model to run:
    needed_raw_cols = {'redCards', 'log_games', 'SkinToneBin_Dark', 'SkinToneMean', 'refNum'}
    needed_raw_cols.update(controls)
    # Add raw names for categorical controls
    if 'position' in df.columns:
        needed_raw_cols.add('position')
    if 'leagueCountry' in df.columns:
        needed_raw_cols.add('leagueCountry')

    needed_raw_cols = [c for c in needed_raw_cols if c in df.columns]
    model_df = df.copy()
    model_df = model_df.dropna(subset=needed_raw_cols).reset_index(drop=True)

    if model_df.shape[0] == 0:
        raise ValueError("No observations remain after dropping rows with missing model variables.")

    # Convert refNum to categorical codes for clustering to ensure groups are integer-coded 0..G-1
    groups = pd.Categorical(model_df['refNum']).codes
    # Fit Negative Binomial GLM with offset = log_games
    try:
        fit_res = smf.glm(formula=formula,
                         data=model_df,
                         family=sm.families.NegativeBinomial(),
                         offset=model_df['log_games']).fit(cov_type='cluster', cov_kwds={'groups': groups})
    except Exception as e:
        # Fall back to Poisson with robust SEs if NegativeBinomial fails
        print('NegativeBinomial failed with error:', e)
        print('Falling back to Poisson with cluster-robust SEs')
        fit_res = smf.glm(formula=formula,
                         data=model_df,
                         family=sm.families.Poisson(),
                         offset=model_df['log_games']).fit(cov_type='cluster', cov_kwds={'groups': groups})

    print(fit_res.summary())

    # Also return a simple incidence rate ratio (IRR) table for key coefficients
    params = fit_res.params
    conf = fit_res.conf_int()
    irr = pd.DataFrame({
        'coef': params,
        'IRR': np.exp(params),
        'IRR_lower': np.exp(conf[0]),
        'IRR_upper': np.exp(conf[1])
    })
    key_mask = [ ('SkinToneBin_Dark' in idx) or ('SkinToneMean' in idx) for idx in irr.index ]
    print('\nIncidence Rate Ratios (IRR) and 95% CI')
    print(irr.loc[key_mask])

    return fit_res