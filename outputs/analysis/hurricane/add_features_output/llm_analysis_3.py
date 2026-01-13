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
    Transform the raw hurricane dataframe into a modeling-ready dataframe.

    Produces these columns used in the model:
      - alldeaths: integer count of deaths (DV)
      - masfem_z: standardized masfem (continuous IV)
      - gender_mf: binary female-name indicator (alternative IV)
      - wind: max wind speed at landfall (control)
      - category: Saffir-Simpson category (control)
      - min: minimum pressure at landfall (control)
      - year_c: mean-centered year (control)
      - source: original source column (categorical control)

    Notes:
      - Drops rows with missing values in variables required for the primary specifications.
      - Keeps raw alldeaths as integer for count modeling (Negative Binomial). If alldeaths is non-integer, it is coerced to integer.
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure required columns exist
    required = ['alldeaths', 'masfem', 'gender_mf', 'wind', 'category', 'min', 'year', 'source']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Coerce alldeaths to numeric (counts). Fill or drop non-sensical values (negative)
    df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce')
    # Remove negative death counts if any
    df.loc[df['alldeaths'] < 0, 'alldeaths'] = np.nan

    # Coerce numeric controls
    df['masfem'] = pd.to_numeric(df['masfem'], errors='coerce')
    df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce')
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')
    df['category'] = pd.to_numeric(df['category'], errors='coerce')
    df['min'] = pd.to_numeric(df['min'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    # Keep the source column as a categorical for modeling (string or category)
    # Convert before dropna so that missing sources become 'unknown' rather than being dropped
    df['source'] = df['source'].astype(str)
    df.loc[df['source'].isin(['nan', 'None', 'NoneType']) | df['source'].isna(), 'source'] = 'unknown'
    df['source'] = df['source'].fillna('unknown')

    # Drop rows with missing values in core modeling variables (complete-case for primary specs)
    df = df.dropna(subset=['alldeaths', 'masfem', 'gender_mf', 'wind', 'category', 'min', 'year', 'source'])

    # Ensure alldeaths is integer (counts). If fractional, round to nearest integer.
    # Keep non-negative integers
    df['alldeaths'] = df['alldeaths'].round().astype(int)

    # Standardize masfem to z-scores for interpretability
    masfem_mean = df['masfem'].mean()
    masfem_std = df['masfem'].std(ddof=0) if df['masfem'].std(ddof=0) != 0 else 1.0
    df['masfem_z'] = (df['masfem'] - masfem_mean) / masfem_std

    # Mean-center year to aid interpretation of intercept
    df['year_c'] = df['year'] - df['year'].mean()

    # Final column list note: model will use alldeaths, masfem_z (primary IV) and gender_mf (alternative),
    # wind, category, min, year_c, and source.
    return df

# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit count regression models testing whether more-feminine hurricane names predict fewer precautionary measures
    proxied by fatalities (alldeaths).

    Primary specification: Negative binomial regression with continuous masfem_z.
    Robustness: Negative binomial with binary gender_mf.

    Returns a dictionary with fitted models and summaries.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure transform has been applied (expect required columns to exist)
    required = ['alldeaths', 'masfem_z', 'gender_mf', 'wind', 'category', 'min', 'year_c', 'source']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing}")

    # Use a formula that includes the categorical source as fixed effects (reference chosen automatically)
    formula_cont = 'alldeaths ~ masfem_z + wind + category + min + year_c + C(source)'
    formula_bin = 'alldeaths ~ gender_mf + wind + category + min + year_c + C(source)'

    # Fit Negative Binomial via GLM (handles overdispersion vs Poisson)
    # Primary model (continuous femininity)
    model_nb_cont = smf.glm(formula=formula_cont, data=df, family=sm.families.NegativeBinomial()).fit()
    # Robust covariance version (HC3)
    model_nb_cont_robust = smf.glm(formula=formula_cont, data=df, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')

    # Robustness model (binary female name)
    model_nb_bin = smf.glm(formula=formula_bin, data=df, family=sm.families.NegativeBinomial()).fit()
    model_nb_bin_robust = smf.glm(formula=formula_bin, data=df, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')

    # Prepare simple model summaries (text) and return full result objects for deeper inspection
    results = {
        'nb_cont_model': model_nb_cont,
        'nb_cont_model_robust': model_nb_cont_robust,
        'nb_bin_model': model_nb_bin,
        'nb_bin_model_robust': model_nb_bin_robust,
        'summary_cont_text': model_nb_cont_robust.summary().as_text(),
        'summary_bin_text': model_nb_bin_robust.summary().as_text()
    }

    # Optionally print brief summaries
    print('Negative Binomial (continuous masfem) — coefficients:')
    print(model_nb_cont_robust.params)
    print('\nNegative Binomial (binary gender_mf) — coefficients:')
    print(model_nb_bin_robust.params)

    return results