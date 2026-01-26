from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/positive_leading_statement_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into the modeling dataframe.
    Produces:
      - masfem_z: z-scored femininity rating (continuous IV)
      - IsFemaleName: binary indicator from gender_mf (alternate IV)
      - log_alldeaths: log(1 + alldeaths) (primary DV)
      - log_ndam15: log(1 + ndam15) (auxiliary outcome for robustness)
      - ensures numeric types for controls and drops rows missing core vars
    """
    df = df.copy()

    # Ensure the core numeric columns exist and coerce types
    numeric_cols = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'elapsedyrs', 'ndam15', 'year']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the primary IV or primary DV or key meteorological controls
    required = ['masfem', 'alldeaths', 'wind', 'min', 'category']
    df = df.dropna(subset=[c for c in required if c in df.columns])

    # Create binary indicator for female name (gender_mf is 0/1 already in dataset, but coerce)
    if 'gender_mf' in df.columns:
        df['IsFemaleName'] = (df['gender_mf'] == 1).astype(int)
    else:
        # if gender_mf missing, create NA-coded column
        df['IsFemaleName'] = np.nan

    # Standardize masfem to z-score for interpretability
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # Dependent variable: log(1 + alldeaths) to reduce skew and handle zeros
    df['log_alldeaths'] = np.log1p(df['alldeaths'].fillna(0))

    # Auxiliary outcome: log(1 + ndam15) — economic damage normalized to 2015 values
    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log1p(df['ndam15'].fillna(0))
    else:
        df['log_ndam15'] = np.nan

    # Keep source as categorical (do not expand to dummies here; handled in modeling via C(source))
    if 'source' in df.columns:
        df['source'] = df['source'].astype('category')
    else:
        df['source'] = 'unknown'

    # Ensure category and elapsedyrs are numeric and drop rows missing them (key controls)
    if 'category' in df.columns:
        df['category'] = pd.to_numeric(df['category'], errors='coerce')
    if 'elapsedyrs' in df.columns:
        df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')

    df = df.dropna(subset=[c for c in ['category', 'elapsedyrs'] if c in df.columns])

    # Final small check: drop rows where masfem_z or log_alldeaths are missing
    df = df.dropna(subset=['masfem_z', 'log_alldeaths'])

    # Return only columns that will be used later (keeps dataframe compact)
    keep_cols = ['ind', 'year', 'name', 'masfem', 'masfem_z', 'gender_mf', 'IsFemaleName', 'min', 'wind', 'category', 'alldeaths', 'log_alldeaths', 'ndam15', 'log_ndam15', 'elapsedyrs', 'source']
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit primary and robustness models to test whether more feminine hurricane names are associated
    with worse outcomes (used here as proxy evidence for fewer precautions):
      - Primary model: OLS on log_alldeaths with robust (HC3) SEs
      - Robustness 1: Poisson GLM on raw alldeaths (count model)
      - Robustness 2: Negative binomial GLM (if it converges)
      
    Returns a dictionary with fitted results objects.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Work on a copy
    df = df.copy()

    # Define a common covariate set. We include C(source) so that source is treated categorically.
    formula_base = 'masfem_z + IsFemaleName + wind + min + category + elapsedyrs + C(source)'

    # Primary: OLS on log-transformed fatalities
    formula_ols = 'log_alldeaths ~ ' + formula_base
    ols_model = smf.ols(formula_ols, data=df)
    ols_res = ols_model.fit(cov_type='HC3')

    # Robustness: Poisson on raw counts (alldeaths). Poisson can be informative for counts.
    formula_pois = 'alldeaths ~ ' + formula_base
    try:
        pois_model = smf.glm(formula_pois, data=df, family=sm.families.Poisson())
        pois_res = pois_model.fit(cov_type='HC3', maxiter=200)
    except Exception as e:
        pois_res = None

    # Robustness: Negative binomial (handles overdispersion)
    try:
        nb_model = smf.glm(formula_pois, data=df, family=sm.families.NegativeBinomial())
        nb_res = nb_model.fit(cov_type='HC3')
    except Exception as e:
        nb_res = None

    # Additional robustness: OLS on economic damage (log_ndam15)
    ols_damage_res = None
    if 'log_ndam15' in df.columns and df['log_ndam15'].notna().sum() > 10:
        try:
            formula_damage = 'log_ndam15 ~ ' + formula_base
            ols_damage_res = smf.ols(formula_damage, data=df).fit(cov_type='HC3')
        except Exception:
            ols_damage_res = None

    results = {
        'ols_log_deaths': ols_res,
        'poisson_deaths': pois_res,
        'negbin_deaths': nb_res,
        'ols_log_damage': ols_damage_res,
        'model_formula': {
            'ols': formula_ols,
            'poisson': formula_pois
        }
    }

    return results


