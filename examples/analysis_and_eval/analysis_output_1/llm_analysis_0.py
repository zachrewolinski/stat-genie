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
    Transformations performed:
    - Ensure relevant columns exist and drop rows missing critical variables for the main analyses.
    - Create standardized femininity score (masfem_z) and masfem_mturk_z (if masfem_mturk available) for robustness checks.
    - Create log-transformed damage (log_ndam15) and log1p deaths (log_alldeaths) for sensitivity analyses.
    - Create a categorical version of category (category_cat) and a centered year (year_centered).
    - Return dataframe containing all columns referenced by the modeling code.
    """
    # Work on a copy
    df = df.copy()

    # Ensure numeric conversions where appropriate
    # (If these conversions fail because of formatting issues they will raise; user can inspect)
    for col in ['alldeaths', 'masfem', 'wind', 'min', 'category', 'ndam15', 'year', 'gender_mf', 'masfem_mturk']:
        if col in df.columns:
            # coerce errors to NaN to allow dropna
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the primary outcome or primary predictor(s)
    required_for_main = ['alldeaths', 'masfem', 'wind', 'min', 'category', 'ndam15', 'year']
    present_required = [c for c in required_for_main if c in df.columns]
    df = df.dropna(subset=present_required)

    # Standardize masfem (z-score) for interpretability
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / (df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0)

    # If masfem_mturk exists, also create its z-score for robustness checks
    if 'masfem_mturk' in df.columns:
        df['masfem_mturk_z'] = (df['masfem_mturk'] - df['masfem_mturk'].mean()) / (df['masfem_mturk'].std(ddof=0) if df['masfem_mturk'].std(ddof=0) != 0 else 1.0)

    # Ensure binary gender indicator is numeric 0/1 if present
    if 'gender_mf' in df.columns:
        df['gender_mf'] = df['gender_mf'].astype(float)

    # Create log-transformed damage (ndam15) for control (highly skewed)
    if 'ndam15' in df.columns:
        # Use log1p to handle zeros safely
        df['log_ndam15'] = np.log1p(df['ndam15'])
    else:
        # If ndam15 not present but ndam is present, use ndam
        if 'ndam' in df.columns:
            df['log_ndam15'] = np.log1p(df['ndam'])

    # Some sensitivity analyses may use log of deaths; create for convenience
    df['log_alldeaths'] = np.log1p(df['alldeaths'])

    # Center year to reduce collinearity and improve interpretability
    if 'year' in df.columns:
        df['year_centered'] = df['year'] - df['year'].median()

    # Make category a categorical column for use in formulas
    if 'category' in df.columns:
        # Keep original numeric category but also create categorical version
        df['category_cat'] = df['category'].astype('category')

    # Final note: keep only the columns required for modeling to avoid accidental usage of other columns
    needed_cols = ['alldeaths', 'masfem_z', 'gender_mf', 'wind', 'min', 'category_cat', 'log_ndam15', 'year_centered', 'log_alldeaths']
    # include masfem_mturk_z if it was created
    if 'masfem_mturk_z' in df.columns:
        needed_cols.append('masfem_mturk_z')

    # Keep any of the needed columns that exist in the df
    needed_cols = [c for c in needed_cols if c in df.columns]

    # Return the dataframe with all original columns plus the new derived ones. Many workflows expect the full df,
    # but ensure the derived columns are present. We'll return the full df (not subset) to be flexible.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Statistical modeling to test whether more feminine hurricane names are associated with fewer fatalities.

    Primary model: Negative binomial GLM with count outcome alldeaths to account for over-dispersion in death counts.
    - DV: alldeaths (count)
    - IV: masfem_z (standardized femininity)
    - Controls: wind, min (pressure), category_cat (categorical), log_ndam15, year_centered

    Robustness checks returned alongside the main model:
    1) Replace masfem_z with binary gender_mf.
    2) OLS on log_alldeaths as an alternative (Gaussian approx after log-transform).

    Returns a dict containing the fitted results objects for downstream inspection.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    results = {}

    # Ensure required columns exist
    required_cols = ['alldeaths', 'masfem_z', 'wind', 'min', 'category_cat', 'log_ndam15', 'year_centered']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataframe for modeling: {missing}")

    # Main specification: Negative Binomial GLM
    formula_nb = 'alldeaths ~ masfem_z + wind + min + C(category_cat) + log_ndam15 + year_centered'
    nb_model = smf.glm(formula=formula_nb, data=df, family=sm.families.NegativeBinomial())
    nb_res = nb_model.fit()
    results['nb_model'] = nb_res

    # Robustness 1: use binary gender_mf instead of masfem_z (if available)
    if 'gender_mf' in df.columns:
        formula_nb_g = 'alldeaths ~ gender_mf + wind + min + C(category_cat) + log_ndam15 + year_centered'
        nb_model_g = smf.glm(formula=formula_nb_g, data=df, family=sm.families.NegativeBinomial())
        nb_res_g = nb_model_g.fit()
        results['nb_model_gender'] = nb_res_g

    # Robustness 2: OLS on log(1+deaths)
    if 'log_alldeaths' in df.columns:
        formula_ols = 'log_alldeaths ~ masfem_z + wind + min + C(category_cat) + log_ndam15 + year_centered'
        ols_res = smf.ols(formula=formula_ols, data=df).fit()
        results['ols_log'] = ols_res

    # Return the results dictionary. Each value is a fitted statsmodels results object; the caller can print summary()
    return results


