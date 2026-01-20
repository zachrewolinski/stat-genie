from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/hurricane/noperturb_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure numeric conversion where needed
    numeric_columns = ['masfem', 'masfem_mturk', 'gender_mf', 'ndam15', 'ndam', 'alldeaths',
                       'wind', 'min', 'category', 'year', 'elapsedyrs']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Rename/duplicate min to avoid shadowing builtins and make variable name explicit
    if 'min' in df.columns:
        df['min_pressure'] = df['min']

    # Drop rows missing the primary variables needed for main analysis
    required_for_main = ['masfem', 'ndam15', 'wind', 'min_pressure', 'category', 'year']
    missing_required = df[required_for_main].isnull().any(axis=1)
    df = df.loc[~missing_required].reset_index(drop=True)

    # Create dependent variable: log-transformed damage (ndam15) to reduce skew
    df['log_ndam15'] = np.log(df['ndam15'].astype(float) + 1.0)

    # Also create log deaths as a secondary outcome for robustness checks
    if 'alldeaths' in df.columns:
        df['log_alldeaths'] = np.log(df['alldeaths'].astype(float) + 1.0)

    # Standardize masfem and masfem_mturk (z-scores)
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1.0)
    else:
        df['masfem_mturk_z'] = np.nan

    # Binary female-coded name indicator (explicitly typed)
    if 'gender_mf' in df.columns:
        # In original data 1 indicates female, 0 male
        df['gender_female'] = df['gender_mf'].astype(int)
    else:
        df['gender_female'] = np.nan

    # Center year to improve interpretability
    df['year_c'] = df['year'] - df['year'].mean()

    # Keep only columns needed for modeling + original identifiers for traceability
    keep_cols = [c for c in df.columns if c in (
        ['ind', 'name', 'year', 'year_c', 'masfem', 'masfem_z', 'masfem_mturk', 'masfem_mturk_z',
         'gender_mf', 'gender_female', 'ndam15', 'log_ndam15', 'alldeaths', 'log_alldeaths',
         'wind', 'min_pressure', 'category', 'elapsedyrs']
    )]

    df = df[keep_cols].copy()

    # Final sanity: drop rows with missing values in the model columns used in the main specification
    model_required = ['masfem_z', 'log_ndam15', 'wind', 'min_pressure', 'category', 'year_c']
    df = df.dropna(subset=model_required).reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """Fit OLS models testing whether more-feminine hurricane names predict larger damages (proxy for fewer precautions).

    Returns a dict with:
      - 'damage_model': OLS result predicting log_ndam15 (primary outcome)
      - 'death_model': OLS result predicting log_alldeaths (secondary robustness outcome; may be NaN if missing)

    Main specification controls for storm intensity (wind, min_pressure, category) and year trends.
    """
    import statsmodels.formula.api as smf

    results = {}

    # Primary model: damage
    formula_damage = 'log_ndam15 ~ masfem_z + gender_female + wind + min_pressure + category + year_c + masfem_mturk_z'
    # Note: masfem_mturk_z included as an additional control/robustness covariate; if missing it will be ignored by patsy only if present.
    # To be robust to missing masfem_mturk_z, ensure column exists in df (transform ensures it exists but may be NaN).

    # Fit model; drop rows with any NaNs in the formula variables (statsmodels will do this automatically but we explicitly subset for clarity)
    dmg_vars = ['log_ndam15', 'masfem_z', 'gender_female', 'wind', 'min_pressure', 'category', 'year_c', 'masfem_mturk_z']
    dmg_df = df[dmg_vars].dropna()
    if dmg_df.shape[0] < 5:
        raise ValueError('Too few observations to fit the damage model after dropping missing values.')

    damage_model = smf.ols(formula_damage, data=dmg_df).fit()
    results['damage_model'] = damage_model

    # Secondary model: fatalities (robustness). Only run if log_alldeaths exists and has non-missing values
    if 'log_alldeaths' in df.columns and df['log_alldeaths'].notna().sum() >= 5:
        death_vars = ['log_alldeaths', 'masfem_z', 'gender_female', 'wind', 'min_pressure', 'category', 'year_c', 'masfem_mturk_z']
        death_df = df[death_vars].dropna()
        if death_df.shape[0] >= 5:
            formula_death = 'log_alldeaths ~ masfem_z + gender_female + wind + min_pressure + category + year_c + masfem_mturk_z'
            death_model = smf.ols(formula_death, data=death_df).fit()
            results['death_model'] = death_model
        else:
            results['death_model'] = None
    else:
        results['death_model'] = None

    # Return the fitted results objects so the caller can inspect .summary(), coefficients, CIs, etc.
    return results


