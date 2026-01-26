from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/replace_and_positive_statement_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataset into analysis-ready columns.

    Produces the following new columns (all appear in the returned dataframe):
      - log_alldeaths: np.log1p(alldeaths)
      - masfem_z: z-scored masfem (perceived femininity)
      - masfem_mturk_z: z-scored masfem_mturk (if present; kept for robustness)
      - gender_female: integer copy of gender_mf (0/1)
      - damage_log: np.log1p(ndam15)
      - year_center: year minus mean(year)

    Also drops rows missing any of the core variables required for the main models
    (alldeaths, masfem, gender_mf, wind, min, category, elapsedyrs, ndam15, year).
    """
    df = df.copy()

    # Required columns for the principal analyses
    required = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'elapsedyrs', 'ndam15', 'year']
    missing_required = [c for c in required if c not in df.columns]
    if len(missing_required) > 0:
        raise ValueError(f"The dataframe is missing required columns for transformation: {missing_required}")

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required).copy()

    # Dependent variable: log-transform deaths to reduce skew and handle zeros
    df['log_alldeaths'] = np.log1p(df['alldeaths'].astype(float))

    # Independent variables: standardize masfem for interpretability
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # If masfem_mturk exists, also create a z-scored version for robustness checks
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1.0)

    # Binary gender indicator (ensure integer 0/1)
    df['gender_female'] = df['gender_mf'].astype(int)

    # Damage control: log-transform normalized damage (ndam15)
    df['damage_log'] = np.log1p(df['ndam15'].astype(float))

    # Center year to ease interpretation of intercept and reduce collinearity with trend
    df['year_center'] = df['year'].astype(float) - df['year'].astype(float).mean()

    # Ensure numeric types for controls
    numeric_controls = ['wind', 'min', 'category', 'elapsedyrs']
    for col in numeric_controls:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # After conversions, drop any rows that inadvertently became NA
    model_cols = ['log_alldeaths', 'masfem_z', 'gender_female', 'wind', 'min', 'category', 'elapsedyrs', 'damage_log', 'year_center']
    # If masfem_mturk_z exists, include it in the returned dataframe but not required
    df = df.dropna(subset=model_cols).copy()

    # Return the full dataframe (not subset), including new columns for diagnostics/robustness
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit primary and robustness models testing whether more feminine hurricane names
    are associated with fewer precautionary measures as proxied by higher fatalities.

    Primary specification:
      - Outcome: log_alldeaths (log(1 + alldeaths))
      - Key predictors: masfem_z (continuous femininity), gender_female (binary)
      - Controls: wind, min (pressure), category, elapsedyrs, damage_log, year_center
      - Estimation: OLS on log outcome with robust (HC3) SEs

    Robustness/specification checks:
      - Negative binomial GLM on raw counts (alldeaths) to account for count nature and overdispersion
      - Poisson GLM as sensitivity

    Returns a dict with fitted model results objects for programmatic inspection.
    """
    # Columns used in models
    X_cols = ['masfem_z', 'gender_female', 'wind', 'min', 'category', 'elapsedyrs', 'damage_log', 'year_center']

    # Drop rows with missing values in the model columns (safety)
    df_model = df.dropna(subset=X_cols + ['log_alldeaths', 'alldeaths']).copy()

    # Design matrix
    X = sm.add_constant(df_model[X_cols])

    # 1) Linear model on log deaths (primary)
    y_log = df_model['log_alldeaths']
    ols_res = sm.OLS(y_log, X).fit(cov_type='HC3')

    # 2) Negative binomial on counts (robust to overdispersion)
    y_counts = df_model['alldeaths'].astype(float)
    try:
        nb_res = sm.GLM(y_counts, X, family=sm.families.NegativeBinomial()).fit()
    except Exception as e:
        # If NB fails to converge or raise error, capture the exception and set nb_res to None
        nb_res = None
        print('NegativeBinomial model failed:', e)

    # 3) Poisson (sensitivity)
    try:
        poisson_res = sm.GLM(y_counts, X, family=sm.families.Poisson()).fit()
    except Exception as e:
        poisson_res = None
        print('Poisson model failed:', e)

    # Print brief summaries for quick inspection
    print('\n===== OLS on log(alldeaths + 1) (HC3 SE) =====')
    print(ols_res.summary())
    if nb_res is not None:
        print('\n===== Negative Binomial on alldeaths =====')
        print(nb_res.summary())
    if poisson_res is not None:
        print('\n===== Poisson on alldeaths =====')
        print(poisson_res.summary())

    # Return model result objects for programmatic access
    return {
        'ols': ols_res,
        'negative_binomial': nb_res,
        'poisson': poisson_res,
        'model_columns': X_cols,
        'n_obs': df_model.shape[0]
    }


