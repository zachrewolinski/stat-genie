from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/projects/binyu/hao_huang/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/hurricane/data.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataset into the analysis-ready dataframe.
    Produces standardized femininity variables, centers year, ensures categorical source,
    and constructs a log outcome for OLS robustness.
    Required output columns (used by models):
      - alldeaths, masfem_z, gender_mf, wind, category, min, elapsedyrs, year_c, source, masfem_mturk_z, log_alldeaths
    """
    df = df.copy()

    # Ensure numeric columns are numeric
    numeric_cols = ['alldeaths', 'masfem', 'masfem_mturk', 'gender_mf', 'wind', 'category', 'min', 'elapsedyrs', 'year']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Ensure source is categorical and fill missing source with 'unknown'
    if 'source' in df.columns:
        df['source'] = df['source'].fillna('unknown').astype('category')
    else:
        df['source'] = pd.Categorical(['unknown'] * len(df))

    # Drop rows missing the key dependent or independent variables
    df = df.dropna(subset=['alldeaths', 'masfem'])

    # Standardize masfem (primary IV) and masfem_mturk (alternative rating) for interpretability
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1)
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1)
    else:
        df['masfem_mturk_z'] = np.nan

    # Ensure binary gender_mf is 0/1
    if 'gender_mf' in df.columns:
        df['gender_mf'] = df['gender_mf'].astype(float).fillna(0).astype(int)
    else:
        df['gender_mf'] = 0

    # Center year to aid model stability (year_c = year - mean(year))
    if 'year' in df.columns:
        df['year_c'] = df['year'] - df['year'].mean()
    else:
        df['year_c'] = 0

    # Create a logged outcome for OLS robustness (log1p to handle zeros)
    df['log_alldeaths'] = np.log1p(df['alldeaths'].fillna(0))

    # Keep only rows with necessary control variables available; for GLM we allow some missing in masfem_mturk_z
    required_for_model = ['alldeaths', 'masfem_z', 'gender_mf', 'wind', 'category', 'min', 'elapsedyrs', 'year_c', 'source']
    missing_req = [c for c in required_for_model if c not in df.columns]
    if missing_req:
        raise ValueError(f"Required columns missing from input df: {missing_req}")

    df = df.dropna(subset=['wind', 'category', 'min', 'elapsedyrs'])

    # Final check / reorder to make outputs clear
    keep_cols = ['alldeaths', 'log_alldeaths', 'masfem', 'masfem_z', 'masfem_mturk_z', 'gender_mf', 'wind', 'category', 'min', 'elapsedyrs', 'year', 'year_c', 'source', 'name', 'ind']
    cols_present = [c for c in keep_cols if c in df.columns]
    df = df[cols_present]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a set of models to test whether more-feminine hurricane names predict differences in fatalities
    after controlling for storm intensity and other covariates.

    Models fit:
      1) Poisson GLM (for count outcome)
      2) Negative Binomial GLM (accounts for overdispersion)
      3) OLS on log1p(alldeaths) as a robustness check

    Returns a dictionary of fitted model result objects.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Ensure we're using the transformed data columns defined above
    required = ['alldeaths', 'masfem_z', 'gender_mf', 'wind', 'category', 'min', 'elapsedyrs', 'year_c', 'source', 'masfem_mturk_z', 'log_alldeaths']
    for c in ['alldeaths', 'masfem_z', 'gender_mf', 'wind', 'category', 'min', 'elapsedyrs', 'year_c', 'source']:
        if c not in df.columns:
            raise ValueError(f"Required column for modeling missing: {c}")

    # Base formula: primary IV masfem_z, include binary gender label and intensity controls, plus source fixed effects and masfem_mturk_z as measurement control
    formula = 'alldeaths ~ masfem_z + gender_mf + wind + category + min + elapsedyrs + year_c + masfem_mturk_z + C(source)'

    # Poisson GLM
    poisson_model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()

    # Compute dispersion for Poisson (pearson chi2 / df_resid). If >>1 suggests overdispersion.
    try:
        poisson_dispersion = poisson_model.pearson_chi2 / poisson_model.df_resid
    except Exception:
        poisson_dispersion = None

    # Negative binomial GLM to account for overdispersion
    negbin_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial()).fit()

    # OLS on log1p(alldeaths) as a robustness check (continuous approximation)
    ols_formula = 'log_alldeaths ~ masfem_z + gender_mf + wind + category + min + elapsedyrs + year_c + masfem_mturk_z + C(source)'
    ols_model = smf.ols(formula=ols_formula, data=df).fit()

    # Pack useful outputs: models themselves and diagnostic numbers
    results = {
        'poisson_model': poisson_model,
        'poisson_dispersion': poisson_dispersion,
        'negbin_model': negbin_model,
        'ols_log_model': ols_model
    }

    return results


