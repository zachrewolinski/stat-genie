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
    Transform the raw hurricane dataframe into the analysis-ready dataframe.
    Produces the following added/cleaned columns used in the models:
      - masfem_z: z-scored masfem (higher = more feminine name)
      - gender_female: integer 0/1 version of gender_mf
      - year_centered: year minus mean(year)
      - log_alldeaths_p1: log(1 + alldeaths) for OLS sensitivity checks
      - log_ndam15_p1: log(1 + ndam15) for damage-based sensitivity checks

    Also ensures alldeaths is integer and drops rows missing core variables.
    """
    df = df.copy()

    # Columns we expect and will use; keep other columns but require these be present
    required = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'year', 'elapsedyrs', 'ndam15']

    # Some datasets may have missing columns; raise informative error if so
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input dataframe is missing required columns: {missing_cols}")

    # Drop rows missing the core outcome or primary IV or core severity controls
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind', 'min', 'category', 'year'])

    # Ensure numeric types
    for col in ['alldeaths', 'masfem', 'gender_mf', 'wind', 'min', 'category', 'year', 'elapsedyrs', 'ndam15']:
        # coerce to numeric where possible
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows that became NA after coercion
    df = df.dropna(subset=['alldeaths', 'masfem', 'wind', 'min', 'category', 'year'])

    # Ensure alldeaths is integer non-negative
    df['alldeaths'] = df['alldeaths'].astype(int)
    df.loc[df['alldeaths'] < 0, 'alldeaths'] = 0

    # Standardize masfem for interpretability
    masfem_mean = df['masfem'].mean()
    masfem_std = df['masfem'].std(ddof=0)
    if masfem_std == 0 or np.isnan(masfem_std):
        # fallback: avoid division by zero
        df['masfem_z'] = 0.0
    else:
        df['masfem_z'] = (df['masfem'] - masfem_mean) / masfem_std

    # Binary female name indicator
    df['gender_female'] = df['gender_mf'].fillna(0).astype(int)

    # Center year to aid interpretation / numeric stability
    df['year_centered'] = df['year'] - df['year'].mean()

    # Add log-transformed versions for OLS sensitivity checks
    df['log_alldeaths_p1'] = np.log1p(df['alldeaths'])
    df['log_ndam15_p1'] = np.log1p(df['ndam15'].fillna(0))

    # Optionally drop extreme outliers on severity measures (commented out; user can enable if desired)
    # df = df[df['wind'] <= df['wind'].quantile(0.995)]

    # Final sanity check: small sample may remain; keep as-is
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fits a set of regressions testing whether more-feminine hurricane names are associated
    with higher fatalities (the hypothesized downstream consequence of reduced precautions).

    Primary specification: Negative Binomial GLM (counts, overdispersion) predicting alldeaths.
    Controls: wind, min (pressure), category, year_centered, elapsedyrs.

    Robustness checks included:
      - Replace continuous masfem (masfem_z) with binary gender_female
      - OLS on log(1 + alldeaths) for an alternative estimator
      - Predicting log(1 + ndam15) (economic damage) as a secondary outcome

    Returns a dict of fitted model result objects.
    """
    results = {}
    # Predictor set for main model using masfem_z
    predictors = ['masfem_z', 'wind', 'min', 'category', 'year_centered', 'elapsedyrs']
    X = df[predictors].copy()
    X = sm.add_constant(X)
    y = df['alldeaths']

    # Fit Negative Binomial via GLM (handles counts with overdispersion)
    try:
        nb_masfem = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
        results['nb_masfem'] = nb_masfem
    except Exception as e:
        # fallback: try statsmodels discrete NegativeBinomial
        try:
            nb2 = sm.NegativeBinomial(y, X).fit(disp=False)
            results['nb_masfem'] = nb2
        except Exception as e2:
            results['nb_masfem_error'] = str(e) + ' | fallback error: ' + str(e2)

    # Alternative: binary gender indicator instead of masfem_z
    predictors_gender = ['gender_female', 'wind', 'min', 'category', 'year_centered', 'elapsedyrs']
    Xg = df[predictors_gender].copy()
    Xg = sm.add_constant(Xg)
    try:
        nb_gender = sm.GLM(y, Xg, family=sm.families.NegativeBinomial()).fit()
        results['nb_gender'] = nb_gender
    except Exception as e:
        try:
            nbg2 = sm.NegativeBinomial(y, Xg).fit(disp=False)
            results['nb_gender'] = nbg2
        except Exception as e2:
            results['nb_gender_error'] = str(e) + ' | fallback error: ' + str(e2)

    # Sensitivity: OLS on log(1 + alldeaths)
    ols_preds = ['masfem_z', 'wind', 'min', 'category', 'year_centered', 'elapsedyrs']
    Xo = df[ols_preds].copy()
    Xo = sm.add_constant(Xo)
    y_log = df['log_alldeaths_p1']
    try:
        ols_model = sm.OLS(y_log, Xo).fit()
        results['ols_log_alldeaths'] = ols_model
    except Exception as e:
        results['ols_log_alldeaths_error'] = str(e)

    # Sensitivity: effect on economic damage (log ndam15 + 1)
    if 'ndam15' in df.columns:
        Xd = df[['masfem_z', 'wind', 'min', 'category', 'year_centered', 'elapsedyrs']].copy()
        Xd = sm.add_constant(Xd)
        y_dam = df['log_ndam15_p1']
        try:
            ols_dam = sm.OLS(y_dam, Xd).fit()
            results['ols_log_ndam15'] = ols_dam
        except Exception as e:
            results['ols_log_ndam15_error'] = str(e)

    # Return results dict; calling code can inspect summaries, coefficients, p-values, etc.
    return results


