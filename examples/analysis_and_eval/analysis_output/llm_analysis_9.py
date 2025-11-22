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
    Clean and derive variables for modeling the relationship between hurricane name femininity
    and fatalities (proxy for precautionary behavior).

    Returns a dataframe that includes the following columns used in the model:
      - alldeaths: integer count of fatalities (DV)
      - masfem_z: standardized (z-scored) femininity index (IV)
      - gender_female: binary 0/1 female-name indicator (IV)
      - wind: max wind speed at landfall (control)
      - category: Saffir-Simpson category (control)
      - min: minimum central pressure at landfall (control)
      - log_ndam15: log(ndam15 + 1) (control)
      - year_centered: year demeaned (control)
      - elapsedyrs: elapsed years (control)
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'category', 'min', 'ndam15', 'year', 'elapsedyrs']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for transform: {missing}")

    # Drop rows missing key variables required for the main analysis
    df = df.dropna(subset=['alldeaths', 'masfem', 'gender_mf', 'wind', 'category', 'min', 'ndam15', 'year'])

    # DV: ensure fatalities are integer counts (some datasets may have floats)
    # Keep zeros; negative or invalid values should be set to NaN and dropped if any
    df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce')
    df = df[df['alldeaths'] >= 0]
    df['alldeaths'] = df['alldeaths'].astype(int)

    # IV: femininity index — standardize (z-score) for interpretability
    df['masfem'] = pd.to_numeric(df['masfem'], errors='coerce')
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # IV: binary female name indicator — coerce to 0/1 integer
    # Original column gender_mf is 0/1; create a clean column named gender_female
    df['gender_female'] = df['gender_mf'].astype(int)

    # Controls: numeric coercion and simple transformations
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    df['category'] = pd.to_numeric(df['category'], errors='coerce')
    df['min'] = pd.to_numeric(df['min'], errors='coerce')
    df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')
    df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    # Log-transform damage (ndam15) for skew reduction; add 1 to handle zeros
    df['log_ndam15'] = np.log(df['ndam15'].clip(lower=0) + 1)

    # Center year to aid interpretation and reduce collinearity with intercept
    df['year_centered'] = df['year'] - df['year'].mean()

    # Final subset: keep only rows without NA in the final set of model columns
    final_cols = ['alldeaths', 'masfem_z', 'gender_female', 'wind', 'category', 'min', 'log_ndam15', 'year_centered', 'elapsedyrs']
    df = df.dropna(subset=final_cols)

    # Return only the final columns needed for modeling (but keep original index)
    return df[final_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fits a negative binomial regression predicting hurricane fatalities (alldeaths) from
    name femininity and controls. Also fits a robustness OLS model predicting logged damage.

    Returns a dict with fitted results objects:
      - 'nb_model': negative binomial model with robust covariance (HC3)
      - 'ols_damage': OLS on log_ndam15 (robust covariance)
    """
    # Prepare design matrix
    model_cols = ['masfem_z', 'gender_female', 'wind', 'category', 'min', 'log_ndam15', 'year_centered', 'elapsedyrs']
    X = df[model_cols]
    X = sm.add_constant(X, has_constant='add')
    y_counts = df['alldeaths']

    # Fit Negative Binomial via GLM (allows modeling overdispersed counts).
    # If the statsmodels version does not support families.NegativeBinomial for GLM,
    # an alternative is to use statsmodels.discrete.discrete_model.NegativeBinomial.
    try:
        nb_family = sm.families.NegativeBinomial()
        nb_model = sm.GLM(y_counts, X, family=nb_family).fit()
        # obtain robust (heteroskedasticity-consistent) covariance estimates
        nb_model_robust = nb_model.get_robustcov_results(cov_type='HC3')
    except Exception:
        # Fallback to discrete NegativeBinomial model if GLM NegativeBinomial isn't available
        from statsmodels.discrete.discrete_model import NegativeBinomial
        nb_model = NegativeBinomial(y_counts, X).fit(disp=False)
        # discrete model has its own robustcov method
        nb_model_robust = nb_model.get_robustcov_results(cov_type='HC3')

    # Robustness: model log damage as a continuous outcome (OLS) to see whether name femininity
    # is associated with lower economic damage (another proxy for precaution/exposure).
    y_log_damage = df['log_ndam15']
    ols_model = sm.OLS(y_log_damage, X).fit(cov_type='HC3')

    # Return fitted models (robust result wrappers where applicable)
    return {
        'nb_model': nb_model_robust,
        'ols_damage': ols_model
    }


