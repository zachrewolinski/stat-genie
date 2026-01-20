from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/soccer/anonymize_output/soccer.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying original
    df = df.copy()

    # Rename commonly used original columns to clearer names
    # Keep original feature names in case some are missing in other datasets
    # Columns mapped:
    # feature1 -> player_id (short name)
    # feature5 -> birthdate (dd.mm.yyyy)
    # feature8 -> position
    # feature9 -> n_matches (number of matches in dyad)
    # feature13 -> goals
    # feature14 -> yellow_cards
    # feature15 -> yellow_red (not used but keep)
    # feature16 -> red_cards
    # feature18, feature19 -> rater skin scores (normalized to 1)
    # feature20 -> referee_id
    # feature21 -> referee_country_id
    # feature22 -> implicit_bias
    # feature25 -> explicit_bias

    col_map = {
        'feature1': 'player_id',
        'feature5': 'birthdate',
        'feature8': 'position',
        'feature9': 'n_matches',
        'feature13': 'goals',
        'feature14': 'yellow_cards',
        'feature15': 'yellow_red',
        'feature16': 'red_cards',
        'feature18': 'rater1_skin',
        'feature19': 'rater2_skin',
        'feature20': 'referee_id',
        'feature21': 'referee_country_id',
        'feature22': 'implicit_bias',
        'feature25': 'explicit_bias'
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure numeric columns are numeric
    for col in ['n_matches', 'goals', 'yellow_cards', 'yellow_red', 'red_cards', 'referee_id', 'referee_country_id', 'implicit_bias', 'explicit_bias']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Parse birthdate and compute age at season midpoint (use 2013-01-01 as season midpoint reference)
    if 'birthdate' in df.columns:
        # birthdate format dd.mm.yyyy according to schema
        df['birthdate'] = pd.to_datetime(df['birthdate'], format='%d.%m.%Y', errors='coerce')
        season_mid = pd.Timestamp('2013-01-01')
        df['Age'] = (season_mid - df['birthdate']).dt.days / 365.25

    # Compute mean skin score from the two raters
    df['rater1_skin'] = pd.to_numeric(df.get('rater1_skin'), errors='coerce')
    df['rater2_skin'] = pd.to_numeric(df.get('rater2_skin'), errors='coerce')
    df['SkinScore'] = df[['rater1_skin', 'rater2_skin']].mean(axis=1)

    # Create a binary skin tone indicator focusing on extremes to answer the research question
    # 1 = Dark (mean >= 0.75), 0 = Light (mean <= 0.25), intermediate values -> set to NaN and drop
    df['SkinDark'] = np.where(df['SkinScore'] >= 0.75, 1,
                              np.where(df['SkinScore'] <= 0.25, 0, np.nan))

    # Keep only dyads with clear dark or light ratings (focus on contrast between extremes)
    df = df[df['SkinDark'].notna()]

    # Remove dyads with zero or missing matches (cannot compute exposure offset log(0))
    df = df[df['n_matches'].notna()]
    df = df[df['n_matches'] > 0]

    # Ensure red_cards is integer count and non-missing
    df['red_cards'] = pd.to_numeric(df['red_cards'], errors='coerce').fillna(0).astype(int)

    # Fill or coerce other numeric control variables
    df['goals'] = pd.to_numeric(df['goals'], errors='coerce').fillna(0)
    df['yellow_cards'] = pd.to_numeric(df['yellow_cards'], errors='coerce').fillna(0)
    df['implicit_bias'] = pd.to_numeric(df['implicit_bias'], errors='coerce')
    df['explicit_bias'] = pd.to_numeric(df['explicit_bias'], errors='coerce')

    # Position: keep as-is (categorical). If missing, fill with 'Unknown'
    if 'position' in df.columns:
        df['position'] = df['position'].astype(str).fillna('Unknown')
    else:
        df['position'] = 'Unknown'

    # Keep the final set of columns required for modeling and downstream checks
    keep_cols = [
        'player_id', 'referee_id', 'referee_country_id',
        'n_matches', 'red_cards', 'SkinScore', 'SkinDark',
        'Age', 'goals', 'yellow_cards', 'position',
        'implicit_bias', 'explicit_bias'
    ]
    # Keep only columns that exist in df and drop the rest
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a Poisson regression for red card counts with the number of matches as exposure (offset).
    We estimate clustered robust standard errors by referee_id to account for non-independence of dyads judged by the same referee.

    Model: red_cards ~ SkinDark + Age + goals + yellow_cards + implicit_bias + explicit_bias + C(position)
    Family: Poisson with log link, offset = log(n_matches).

    Returns the robust results object (statsmodels results with clustered SEs).
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = df.copy()

    # Check that required columns exist
    required = ['red_cards', 'n_matches', 'SkinDark', 'referee_id']
    for col in required:
        if col not in df.columns:
            raise ValueError(f'Required column {col} not found in dataframe')

    # Replace any remaining missing numeric covariates with reasonable defaults (or drop rows)
    covariates = ['Age', 'goals', 'yellow_cards', 'implicit_bias', 'explicit_bias']
    for v in covariates:
        if v in df.columns:
            # keep rows where covariate is not missing; alternatively could fill with mean
            df = df[df[v].notna()]

    # Build formula (include categorical position as factor)
    # Only include implicit/explicit bias if present
    extras = []
    if 'implicit_bias' in df.columns:
        extras.append('implicit_bias')
    if 'explicit_bias' in df.columns:
        extras.append('explicit_bias')

    # Always include Age, goals, yellow_cards if present
    for v in ['Age', 'goals', 'yellow_cards']:
        if v in df.columns:
            extras.insert(0, v)  # ensure these appear early in the formula

    # Make sure 'position' column exists for categorical control
    if 'position' in df.columns:
        extras.append('C(position)')

    rhs = ' + '.join(['SkinDark'] + extras)
    formula = f'red_cards ~ {rhs}'

    # Fit Poisson GLM with offset = log(n_matches)
    offset = np.log(df['n_matches'].astype(float))
    poisson_model = smf.glm(formula=formula, data=df, family=sm.families.Poisson(), offset=offset)
    fitted = poisson_model.fit()

    # Obtain cluster-robust covariance by referee_id
    # Use get_robustcov_results to get clustered SEs
    try:
        robust_res = fitted.get_robustcov_results(cov_type='cluster', groups=df['referee_id'])
    except Exception:
        # Fallback: if clustering fails, return standard fitted results
        robust_res = fitted

    # For interpretation, compute incidence rate ratios (IRR) and 95% CI for coefficients
    params = robust_res.params
    conf = robust_res.conf_int()
    irr = np.exp(params)
    irr_ci_lower = np.exp(conf[0])
    irr_ci_upper = np.exp(conf[1])

    # Assemble a small summary DataFrame for convenient inspection
    irr_table = pd.DataFrame({
        'coef': params,
        'IRR': irr,
        'IRR_ci_lower': irr_ci_lower,
        'IRR_ci_upper': irr_ci_upper
    })

    # Attach the IRR table to the results object for easy access (non-standard but convenient)
    robust_res.irr_table = irr_table

    return robust_res


