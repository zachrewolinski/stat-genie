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
    """
    Transform the raw hurricane dataframe into the analysis-ready dataframe.

    Produces the following key columns used in modelling:
      - log_alldeaths: np.log1p(alldeaths)
      - log_ndam15: np.log1p(ndam15)
      - masfem_z: standardized masfem score (z-score)
      - gender_mf: ensured integer 0/1
      - Intensity: composite z-scored index of (wind, category, inverse min pressure)
      - year_c: centered year
      - elapsedyrs: passed through

    Keeps rows with non-missing values on the variables required for downstream models.
    """
    df = df.copy()

    # Ensure required numeric columns exist
    required_cols = ['masfem', 'gender_mf', 'alldeaths', 'ndam15', 'wind', 'min', 'category', 'year', 'elapsedyrs']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for transform: {missing}")

    # Drop rows with missing values in key variables used to compute transforms
    df = df.dropna(subset=['masfem', 'gender_mf', 'alldeaths', 'ndam15', 'wind', 'min', 'category', 'year', 'elapsedyrs'])

    # Dependent variables (log-transform to reduce skew)
    df['log_alldeaths'] = np.log1p(df['alldeaths'].astype(float))
    df['log_ndam15'] = np.log1p(df['ndam15'].astype(float))

    # Independent variables
    # Standardize masfem to z-score (population-style ddof=0)
    masfem_mean = df['masfem'].mean()
    masfem_std = df['masfem'].std(ddof=0)
    if masfem_std == 0 or np.isnan(masfem_std):
        df['masfem_z'] = 0.0
    else:
        df['masfem_z'] = (df['masfem'] - masfem_mean) / masfem_std

    # Ensure gender_mf is integer 0/1
    df['gender_mf'] = df['gender_mf'].astype(int)

    # Construct an Intensity index that combines wind (higher -> stronger), category (higher -> stronger), and inverse min pressure (lower pressure -> stronger)
    # We z-score each component then average them. Handle constant columns safely.
    def safe_z(series: pd.Series) -> pd.Series:
        s = series.astype(float)
        m = s.mean()
        sd = s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=s.index)
        return (s - m) / sd

    z_wind = safe_z(df['wind'])
    z_cat = safe_z(df['category'])
    # min: lower pressure means stronger storm -> invert
    z_inv_min = safe_z(-df['min'].astype(float))

    df['Intensity'] = (z_wind + z_cat + z_inv_min) / 3.0

    # Time controls
    df['year_c'] = df['year'].astype(float) - df['year'].astype(float).mean()
    # elapsedyrs is already present; ensure numeric
    df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')

    # Final: drop any rows that became NA due to coercion
    df = df.dropna(subset=['masfem_z', 'gender_mf', 'log_alldeaths', 'log_ndam15', 'Intensity', 'year_c', 'elapsedyrs'])

    # Keep only columns needed for analysis plus some identifiers for traceability
    cols_to_keep = ['ind', 'year', 'name', 'masfem', 'masfem_z', 'gender_mf', 'alldeaths', 'log_alldeaths', 'ndam15', 'log_ndam15', 'wind', 'min', 'category', 'Intensity', 'year_c', 'elapsedyrs', 'source']
    available = [c for c in cols_to_keep if c in df.columns]
    df = df[available]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fits two models testing whether more feminine hurricane names are associated with outcomes
    consistent with reduced precautionary behavior / perceived threat.

    Models:
      1) Negative binomial regression predicting raw alldeaths (counts) with name femininity and controls.
      2) OLS regression predicting log-transformed economic damage (log_ndam15) as a robustness check.

    Returns a dict with fitted results objects for both models.
    """
    # Required columns
    required = ['masfem_z', 'gender_mf', 'Intensity', 'year_c', 'elapsedyrs', 'alldeaths', 'log_ndam15']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Drop rows with NA in model variables
    df_model = df.dropna(subset=required).copy()

    # Define predictors
    X_cols = ['masfem_z', 'gender_mf', 'Intensity', 'year_c', 'elapsedyrs']
    X = df_model[X_cols].astype(float)
    X = sm.add_constant(X)

    results = {}

    # 1) Negative binomial for alldeaths (count outcome, overdispersed)
    try:
        nb_model = sm.GLM(df_model['alldeaths'].astype(float), X, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
        results['nb_model'] = nb_model
    except Exception as e:
        # If GLM NB fails, fall back to Poisson with robust SE (still informative)
        poisson_model = sm.GLM(df_model['alldeaths'].astype(float), X, family=sm.families.Poisson()).fit(cov_type='HC3')
        results['nb_model'] = poisson_model
        results['nb_model_warning'] = f"NegativeBinomial failed and Poisson was used as fallback. Original error: {e}"

    # 2) OLS for log-transformed economic damage (continuous skewed -> log transform used)
    ols_model = sm.OLS(df_model['log_ndam15'].astype(float), X).fit(cov_type='HC3')
    results['ols_model'] = ols_model

    # For convenient summary printing in downstream scripts, include a small textual summary
    results['summary'] = {
        'nb_params': results['nb_model'].params.to_dict(),
        'nb_pvalues': results['nb_model'].pvalues.to_dict(),
        'ols_params': results['ols_model'].params.to_dict(),
        'ols_pvalues': results['ols_model'].pvalues.to_dict()
    }

    return results


