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
    Transform the raw hurricane dataset into analysis-ready columns.

    New columns added (and used in modeling):
      - log_alldeaths: log(alldeaths + 1)
      - masfem_z: z-scored masfem ratings (primary IV)
      - masfem_mturk_z: z-scored masfem_mturk (alternative continuous IV) if present
      - min_z: inverted-and-z-scored minimum pressure (higher => more severe)
      - wind_z, category_z: z-scored wind and category
      - StormSeverity: average of wind_z, min_z, category_z (composite severity control)
      - year_cent: centered year

    Rows with missing values in key variables are dropped.
    """
    df = df.copy()

    # Ensure numeric columns are numeric where expected
    for col in ['masfem', 'gender_mf', 'alldeaths', 'wind', 'category', 'min', 'year']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing core columns required for the main analyses
    required = [c for c in ['masfem', 'gender_mf', 'alldeaths', 'wind', 'category', 'min', 'year'] if c in df.columns]
    df = df.dropna(subset=required)

    # Dependent variable: raw counts and log transform to reduce skew
    df['alldeaths'] = df['alldeaths'].astype(float)
    df['log_alldeaths'] = np.log(df['alldeaths'] + 1)

    # Independent variable(s): standardize continuous masfem
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # If masfem_mturk present, also create a standardized version (alternative IV)
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk'] = pd.to_numeric(df['masfem_mturk'], errors='coerce')
        df = df.dropna(subset=['masfem_mturk'])
        df['masfem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1.0)

    # Standardize and combine objective storm severity indicators
    # Invert min (pressure): lower pressure -> more severe; so -min aligns direction with wind and category
    df['wind_z'] = (df['wind'] - df['wind'].mean()) / (df['wind'].std(ddof=0) if df['wind'].std(ddof=0) != 0 else 1.0)
    df['min_z'] = -1.0 * (df['min'] - df['min'].mean()) / (df['min'].std(ddof=0) if df['min'].std(ddof=0) != 0 else 1.0)
    df['category_z'] = (df['category'] - df['category'].mean()) / (df['category'].std(ddof=0) if df['category'].std(ddof=0) != 0 else 1.0)

    # Composite severity: average of z-scored indicators
    df['StormSeverity'] = df[['wind_z', 'min_z', 'category_z']].mean(axis=1)

    # Time control: center year
    df['year_cent'] = df['year'] - df['year'].mean()

    # Ensure binary gender_mf is numeric 0/1
    df['gender_mf'] = df['gender_mf'].astype(int)

    # Keep only columns needed for downstream modeling plus originals for inspection
    keep_cols = list(df.columns)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit statistical models to assess whether more feminine hurricane names are associated with
    different observed human consequences (used here as proxies for precautionary behavior):
      - Negative binomial GLM on alldeaths (count outcome, robust to overdispersion)
      - OLS on log_alldeaths (log-transformed outcome, alternative specification)

    The primary independent variable is masfem_z (standardized femininity rating). Models control
    for objective StormSeverity and year. gender_mf is included as an additional covariate.

    Returns a dictionary with fitted model results objects.
    """
    import statsmodels.api as sm

    # Define model matrix
    X_cols = [c for c in ['masfem_z', 'StormSeverity', 'year_cent', 'gender_mf'] if c in df.columns]
    X = df[X_cols].astype(float)
    X = sm.add_constant(X)

    # Outcome: raw counts of deaths
    y_count = df['alldeaths'].astype(float)

    # Fit a Negative Binomial GLM (handles overdispersion relative to Poisson)
    try:
        nb_model = sm.GLM(y_count, X, family=sm.families.NegativeBinomial()).fit()
    except Exception:
        # Fallback: if NegativeBinomial family unavailable or fails, fit Poisson with robust covariances
        poisson = sm.GLM(y_count, X, family=sm.families.Poisson()).fit()
        nb_model = poisson

    # Alternative specification: OLS on log-transformed deaths
    y_log = df['log_alldeaths'].astype(float)
    ols_model = sm.OLS(y_log, X).fit()

    # Also fit a model using the binary gender_mf alone (for interpretation) if masfem_z present
    secondary = {}
    if 'gender_mf' in X.columns and 'masfem_z' in X.columns:
        X_bin = X.drop(columns=['masfem_z'])
        nb_bin = None
        try:
            nb_bin = sm.GLM(y_count, X_bin, family=sm.families.NegativeBinomial()).fit()
        except Exception:
            nb_bin = sm.GLM(y_count, X_bin, family=sm.families.Poisson()).fit()
        ols_bin = sm.OLS(y_log, X_bin).fit()
        secondary = {'nb_binary_gender': nb_bin, 'ols_binary_gender': ols_bin}

    results = {
        'nb_model': nb_model,
        'ols_model': ols_model,
        'secondary_models': secondary,
        'specification': {
            'X_columns': X_cols,
            'outcome_count': 'alldeaths',
            'outcome_log': 'log_alldeaths'
        }
    }

    return results


