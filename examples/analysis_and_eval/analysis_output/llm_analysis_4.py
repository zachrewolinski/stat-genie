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
    Transform the raw hurricane dataframe into a modeling-ready dataframe.

    Produces standardized (z-scored) continuous predictors with suffix _z and
    creates a log-transformed damage metric log_ndam15 (and its z-score). Also
    creates simple source indicator columns for common source values.

    Final dataframe contains the columns used in the model:
      - Alldeaths (dependent count)
      - masfem_z, gender_mf
      - wind_z, min_z, category_z, log_ndam15_z, year_z
      - source_is_uri, source_is_mwr, source_is_wiki
    """
    df = df.copy()

    # Ensure key columns exist
    required_cols = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'ndam15', 'year', 'source']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Work with a copy and coerce types
    # Keep alldeaths zero rows — do not drop zeros because they are valid counts
    df['Alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce')
    df['masfem'] = pd.to_numeric(df['masfem'], errors='coerce')
    # gender_mf may be 0/1; coerce to numeric
    df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce')
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    df['min'] = pd.to_numeric(df['min'], errors='coerce')
    df['category'] = pd.to_numeric(df['category'], errors='coerce')
    df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    # Drop rows with missing values in core predictors or DV
    df = df.dropna(subset=['Alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'year'])

    # Create log damage (skewed) and fill NA damage with 0
    df['log_ndam15'] = np.log(df['ndam15'].fillna(0) + 1)

    # Standardize continuous predictors for numerical stability and interpretability
    cont_to_z = ['masfem', 'wind', 'min', 'category', 'log_ndam15', 'year']
    for col in cont_to_z:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        # If std is zero (unlikely) avoid division by zero
        if std == 0 or np.isnan(std):
            df[col + '_z'] = 0.0
        else:
            df[col + '_z'] = (df[col] - mean) / std

    # Create simple source indicator variables for common values (case-insensitive match)
    # We use contains to be robust to slight variations (e.g., 'WIKI (http...')
    df['source'] = df['source'].astype(str)
    df['source_is_uri'] = df['source'].str.contains('uri', case=False, na=False).astype(int)
    df['source_is_mwr'] = df['source'].str.contains('MWR', case=False, na=False).astype(int)
    df['source_is_wiki'] = df['source'].str.contains('WIKI', case=False, na=False).astype(int)

    # Keep only the columns needed for downstream modeling (but keep original columns if desired)
    # We return the full dataframe (with new columns) so users can inspect others if needed.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a Negative Binomial generalized linear model predicting fatalities (Alldeaths)
    from name femininity while controlling for storm intensity and other covariates.

    The function uses the z-scored predictors created by transform() and returns
    a fitted results object with robust (HC3) standard errors.

    Model specification (main):
        Alldeaths ~ masfem_z + gender_mf + wind_z + min_z + category_z + log_ndam15_z + year_z + source dummies

    Notes:
    - We exclude one source dummy (source_is_uri) from the design matrix to serve as the reference category.
    - Negative binomial is used to account for over-dispersion in death counts.
    """
    # Ensure required transformed columns exist
    required = ['Alldeaths', 'masfem_z', 'gender_mf', 'wind_z', 'min_z', 'category_z', 'log_ndam15_z', 'year_z', 'source_is_mwr', 'source_is_wiki']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing}. Run transform() first.")

    # Build design matrix. We use source_is_uri as the implicit reference (not included).
    predictors = ['masfem_z', 'gender_mf', 'wind_z', 'min_z', 'category_z', 'log_ndam15_z', 'year_z', 'source_is_mwr', 'source_is_wiki']
    X = df[predictors].copy()
    X = sm.add_constant(X, has_constant='add')
    y = df['Alldeaths']

    # Fit Negative Binomial GLM
    glm_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    res = glm_nb.fit()

    # Obtain robust (HC3) standard errors for inference
    try:
        res_robust = res.get_robustcov_results(cov_type='HC3')
    except Exception:
        # If robust cov can't be computed for some reason, return the plain results
        res_robust = res

    # Print a concise summary for quick inspection (caller may use the returned object for further work)
    print(res_robust.summary())

    return res_robust


