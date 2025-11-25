from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/.venv/lib/python3.10/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into a cleaned dataframe with derived columns
    required for modeling.

    Added / transformed columns (all included in the returned dataframe):
    - alldeaths: ensure numeric (original outcome)
    - log_ndam15: log-transformed property damage (ndam15) used for a robustness model
    - masfem_center: masfem demeaned (centered) to aid interpretability
    - gender_female: binary 0/1 version of gender_mf
    - wind, min, category, elapsedyrs, source: cleaned control variables

    The function drops rows missing the primary variables required for main models.
    """
    # Make a copy to avoid mutating input
    df = df.copy()

    # Ensure key numeric columns are numeric
    numeric_cols = ['masfem', 'masfem_mturk', 'gender_mf', 'min', 'category', 'alldeaths', 'ndam15', 'wind', 'elapsedyrs']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Ensure source is string / categorical
    if 'source' in df.columns:
        df['source'] = df['source'].astype(str)

    # Primary derived columns
    # 1) Center the masfem score for interpretability (mean = 0)
    if 'masfem' in df.columns:
        df['masfem_center'] = df['masfem'] - df['masfem'].mean()
    else:
        df['masfem_center'] = np.nan

    # 2) Binary female indicator from provided gender_mf (already 0/1); make explicit integer
    if 'gender_mf' in df.columns:
        df['gender_female'] = df['gender_mf'].astype('Int64')
    else:
        df['gender_female'] = pd.Series([pd.NA] * len(df))

    # 3) Log-transform of damage (robustness outcome): log( ndam15 + 1 )
    if 'ndam15' in df.columns:
        df['log_ndam15'] = np.log1p(df['ndam15'].astype(float))
    else:
        df['log_ndam15'] = np.nan

    # 4) Ensure alldeaths is numeric and integer-like (count outcome)
    if 'alldeaths' in df.columns:
        df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce').fillna(0).astype(int)
    else:
        df['alldeaths'] = 0

    # 5) Clean control vars (coerce to numeric where appropriate)
    for c in ['wind', 'min', 'category', 'elapsedyrs']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 6) Drop rows missing the primary variables needed for the main models
    required_for_main = ['masfem', 'alldeaths', 'wind', 'min', 'category', 'elapsedyrs', 'source']
    present_required = [c for c in required_for_main if c in df.columns]
    df = df.dropna(subset=present_required)

    # Final: reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit primary and robustness models to test whether more-feminine hurricane names
    are associated with differences in precautionary outcomes (proxied by fatalities
    and property damage).

    Models returned:
    - nb_model: Negative Binomial regression of alldeaths on masfem_center + controls
    - gender_model: Negative Binomial regression of alldeaths on binary gender_female + controls
    - damage_model: OLS regression of log_ndam15 on masfem_center + controls (robustness)

    Each model includes the same set of controls: wind, min (pressure), category,
    elapsedyrs, and categorical source (C(source)). Robust standard errors can be
    obtained from the result objects.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Work on a copy
    df = df.copy()

    # Primary Negative Binomial model: fatalities as count outcome
    # Formula: alldeaths ~ masfem_center + wind + min + category + elapsedyrs + C(source)
    formula_nb = 'alldeaths ~ masfem_center + wind + min + category + elapsedyrs + C(source)'
    try:
        nb_model = smf.glm(formula=formula_nb, data=df, family=sm.families.NegativeBinomial()).fit()
    except Exception as e:
        # If GLM NegativeBinomial fails to converge, fall back to a Poisson with robust SEs
        nb_model = smf.glm(formula=formula_nb, data=df, family=sm.families.Poisson()).fit(cov_type='HC3')

    # Robustness: use binary gender indicator instead of continuous masfem
    formula_gender = 'alldeaths ~ gender_female + wind + min + category + elapsedyrs + C(source)'
    try:
        gender_model = smf.glm(formula=formula_gender, data=df, family=sm.families.NegativeBinomial()).fit()
    except Exception:
        gender_model = smf.glm(formula=formula_gender, data=df, family=sm.families.Poisson()).fit(cov_type='HC3')

    # Robustness: model economic damage (log-transformed) with OLS
    # Use log_ndam15 (log( ndam15 + 1 )) as dependent variable
    formula_damage = 'log_ndam15 ~ masfem_center + wind + min + category + elapsedyrs + C(source)'
    damage_model = smf.ols(formula=formula_damage, data=df).fit(cov_type='HC3')

    # Return fitted results so caller can inspect summaries, coefficients, CIs, etc.
    results = {
        'nb_model': nb_model,
        'gender_model': gender_model,
        'damage_model': damage_model
    }

    return results


