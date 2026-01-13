from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/add_features_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into the modeling dataframe.

    Outputs (columns added or ensured):
      - Masfem_z: standardized (z) version of 'masfem' (continuous femininity rating)
      - LogDeaths: log(alldeaths + 1)
      - LogDamage: log(ndam15 + 1) (secondary outcome)
      - Year_Centered: year - mean(year)

    Keeps original columns used as controls: 'wind', 'category', 'min', 'elapsedyrs', and binary 'gender_mf'.
    Rows with missing values in core variables are dropped.
    """
    # copy to avoid modifying caller's frame
    df = df.copy()

    # Ensure numeric columns are numeric (coerce errors to NaN so we can drop appropriately)
    numeric_cols = ['masfem', 'gender_mf', 'alldeaths', 'ndam15', 'wind', 'category', 'min', 'year', 'elapsedyrs']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing essential columns needed for the main analyses
    required = ['masfem', 'gender_mf', 'alldeaths', 'ndam15', 'wind', 'category', 'min', 'year']
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise KeyError(f"Input dataframe is missing required columns: {missing_required}")

    df = df.dropna(subset=required)

    # Create standardized femininity variable (z-score). Use sample std (pandas default ddof=1).
    df['Masfem_z'] = (df['masfem'] - df['masfem'].mean()) / df['masfem'].std()

    # Dependent variable: log-transformed fatalities (add 1 to keep zeros)
    df['LogDeaths'] = np.log(df['alldeaths'] + 1)

    # Secondary outcome (log-transformed damage in 2015 dollars)
    df['LogDamage'] = np.log(df['ndam15'] + 1)

    # Center year to improve interpretability and reduce collinearity
    df['Year_Centered'] = df['year'] - df['year'].mean()

    # Ensure binary gender_mf is numeric and 0/1
    df['gender_mf'] = df['gender_mf'].astype(int)

    # Keep only columns needed for modeling plus a few helpful original columns
    keep_cols = [
        'Masfem_z', 'masfem', 'gender_mf', 'alldeaths', 'LogDeaths', 'ndam15', 'LogDamage',
        'wind', 'category', 'min', 'year', 'Year_Centered', 'elapsedyrs', 'name', 'ind'
    ]
    # Filter to columns that exist (some datasets may not include every auxiliary column)
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two models that test whether more feminine hurricane names are associated with outcomes
    indicative of lower precaution by the public.

    Models:
      1) Fatalities model (primary): Negative Binomial GLM on raw count alldeaths with IVs
         Masfem_z and gender_mf and controls for storm intensity and time.
      2) Damage model (secondary): OLS on LogDamage with the same predictors.

    Returns a dict with fitted (robust-covariance) results objects for inspection.
    """
    # Required columns for modeling
    exog_vars = ['Masfem_z', 'gender_mf', 'wind', 'category', 'min', 'Year_Centered']
    missing = [c for c in exog_vars if c not in df.columns]
    if missing:
        raise KeyError(f"Dataframe is missing columns required for modeling: {missing}")

    # Design matrix (add constant)
    X = sm.add_constant(df[exog_vars], has_constant='add')

    results = {}

    # 1) Fatalities: Negative Binomial GLM on counts (alldeaths)
    # Use GLM NegativeBinomial and then obtain robust (HC3) covariance estimates.
    y_deaths = df['alldeaths']
    glm_nb = sm.GLM(y_deaths, X, family=sm.families.NegativeBinomial())
    res_nb = glm_nb.fit()
    try:
        res_nb_robust = res_nb.get_robustcov_results(cov_type='HC3')
    except Exception:
        # Fallback if robustcov not available for this result type
        res_nb_robust = res_nb

    results['deaths_model_raw'] = res_nb
    results['deaths_model_robust'] = res_nb_robust

    # 2) Damage: OLS on log-transformed damage (LogDamage)
    if 'LogDamage' in df.columns:
        y_damage = df['LogDamage']
        ols = sm.OLS(y_damage, X)
        res_ols = ols.fit()
        res_ols_robust = res_ols.get_robustcov_results(cov_type='HC3')
        results['damage_model_raw'] = res_ols
        results['damage_model_robust'] = res_ols_robust

    # Return dictionary of results so the caller can inspect summaries, params, and diagnostics
    return results


