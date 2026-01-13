from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/noperturb_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into the final dataframe used for modeling.

    Produces the following new / cleaned columns used in the models:
    - LogDamage: np.log(ndam15 + 1)
    - Deaths: integer alldeaths (kept for auxiliary models)
    - masfem_c: mean-centered masfem score
    - year_c: mean-centered year
    - gender_female: integer copy of gender_mf (0/1)

    Drops rows with missing values in core variables used by the main models.
    """
    df = df.copy()

    # Ensure numeric types where appropriate
    for col in ['masfem', 'masfem_mturk', 'ndam15', 'alldeaths', 'wind', 'min', 'category', 'year', 'elapsedyrs', 'gender_mf']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the core variables needed for the primary analysis
    required = ['masfem', 'ndam15', 'wind', 'min', 'category', 'year']
    df = df.dropna(subset=required)

    # Dependent variable: log of inflation-adjusted damage (ndam15)
    df['LogDamage'] = np.log(df['ndam15'] + 1)

    # Keep count of deaths as an integer column for auxiliary count models
    df['Deaths'] = df['alldeaths'].fillna(0).astype(int)

    # Independent variable: center the masfem score for interpretability
    df['masfem_c'] = df['masfem'] - df['masfem'].mean()

    # Alternative / robustness IV: MTurk ratings (keep raw and centered)
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk'] = pd.to_numeric(df['masfem_mturk'], errors='coerce')

    # Binary female indicator renamed for clarity
    if 'gender_mf' in df.columns:
        df['gender_female'] = df['gender_mf'].astype(int)

    # Year centering to aid interpretation
    df['year_c'] = df['year'] - df['year'].mean()

    # Ensure other controls are numeric and drop rows with missing control data
    controls_needed = ['wind', 'min', 'category', 'elapsedyrs']
    for c in controls_needed:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=controls_needed)

    # Final returned dataframe contains original fields plus derived columns
    # (LogDamage, Deaths, masfem_c, masfem_mturk, gender_female, year_c)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run statistical models testing whether more feminine hurricane names are associated with lower precautionary outcomes,
    operationalized here as lower (logged) inflation-adjusted damages (primary) and as higher/lower deaths (auxiliary).

    Returns a dictionary with:
    - 'ols_log_damage': OLS results for LogDamage
    - 'glm_nb_deaths': GLM Negative Binomial results for raw death counts (robust to overdispersion)
    """
    df = df.copy()

    # Define predictors for both models
    base_controls = ['wind', 'min', 'category', 'masfem_mturk', 'gender_female', 'year_c', 'elapsedyrs']
    # Keep only controls that are present in df
    base_controls = [c for c in base_controls if c in df.columns]

    X_cols = ['masfem_c'] + base_controls

    # Drop rows with missing data in model columns
    model_df = df.dropna(subset=X_cols + ['LogDamage', 'Deaths'])

    X = model_df[X_cols]
    X = sm.add_constant(X)

    # Primary model: OLS on log-damage with robust (HC3) standard errors
    y_damage = model_df['LogDamage']
    ols_res = sm.OLS(y_damage, X).fit(cov_type='HC3')

    # Auxiliary model: Negative Binomial for death counts
    # Use the same covariates; death counts are non-negative integers, often overdispersed.
    y_deaths = model_df['Deaths']
    try:
        glm_nb = sm.GLM(y_deaths, X, family=sm.families.NegativeBinomial()).fit()
    except Exception:
        # If NB fails to converge, fall back to Poisson with robust SEs
        glm_nb = sm.GLM(y_deaths, X, family=sm.families.Poisson()).fit(cov_type='HC3')

    # Return the fitted result objects so the caller can inspect summaries, coefficients, CIs, etc.
    return {
        'ols_log_damage': ols_res,
        'glm_nb_deaths': glm_nb
    }


