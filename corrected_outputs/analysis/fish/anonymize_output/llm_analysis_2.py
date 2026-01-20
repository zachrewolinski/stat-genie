from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/fish/anonymize_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset column names and derive the columns needed for modeling.

    Input dataframe columns expected (original names):
      - feature1: number of fish caught during visit
      - feature2: whether group used live bait (0/1)
      - feature3: whether group had a camper (0/1)
      - feature4: number of adults in group
      - feature5: number of children in group
      - feature6: hours spent in park

    Output dataframe will contain at minimum these columns (exact names used by model):
      - fish_count, live_bait, camper, n_adults, n_children, hours, total_people, fish_per_hour, log_hours
    """
    # Work on a copy
    df = df.copy()

    # Rename features to meaningful column names
    df = df.rename(columns={
        'feature1': 'fish_count',
        'feature2': 'live_bait',
        'feature3': 'camper',
        'feature4': 'n_adults',
        'feature5': 'n_children',
        'feature6': 'hours'
    })

    # Drop rows missing essential variables
    df = df.dropna(subset=['fish_count', 'hours', 'live_bait', 'camper', 'n_adults', 'n_children'])

    # Coerce to numeric types where appropriate
    df['fish_count'] = pd.to_numeric(df['fish_count'], errors='coerce')
    df['hours'] = pd.to_numeric(df['hours'], errors='coerce')
    df['live_bait'] = pd.to_numeric(df['live_bait'], errors='coerce').fillna(0).astype(int)
    df['camper'] = pd.to_numeric(df['camper'], errors='coerce').fillna(0).astype(int)
    df['n_adults'] = pd.to_numeric(df['n_adults'], errors='coerce').fillna(0).astype(int)
    df['n_children'] = pd.to_numeric(df['n_children'], errors='coerce').fillna(0).astype(int)

    # Remove rows with non-positive hours (cannot compute rate / exposure)
    df = df[df['hours'] > 0]

    # Create derived variables
    df['total_people'] = df['n_adults'] + df['n_children']

    # Derived dependent-variable-per-time for descriptive use; model will use fish_count with an offset
    df['fish_per_hour'] = df['fish_count'] / df['hours']

    # log(hours) for convenience (the model uses offset=np.log(df['hours']))
    df['log_hours'] = np.log(df['hours'])

    # Final cleaning: drop any rows where fish_count is missing or negative
    df = df[df['fish_count'].notnull()]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a count regression for fish_count using hours as exposure (offset) to estimate rate of fish caught per hour.

    Modeling strategy:
      1. Fit a Poisson GLM with offset = log(hours) and predictors: live_bait, camper, total_people.
      2. Check dispersion using Pearson chi-square / df_resid. If overdispersion is present (dispersion > 1.5), fit a Negative Binomial GLM.

    Returns:
      - If Poisson adequate: {'chosen_model': 'Poisson', 'poisson_results': poisson_results, 'dispersion': dispersion}
      - If overdispersed: {'chosen_model': 'NegativeBinomial', 'poisson_results': poisson_results, 'nb_results': nb_results, 'dispersion': dispersion}

    The statsmodels result objects are returned so the caller can inspect coefficients, standard errors, confidence intervals, and predicted rates.
    """
    df = df.copy()

    # Required variables
    required = ['fish_count', 'hours', 'live_bait', 'camper', 'total_people']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Define endogenous and exogenous
    y = df['fish_count'].astype(float)
    exog_vars = ['live_bait', 'camper', 'total_people']
    X = sm.add_constant(df[exog_vars].astype(float))

    # Offset (log exposure)
    offset = np.log(df['hours'].astype(float))

    # Fit Poisson model
    poisson_mod = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    poisson_results = poisson_mod.fit()

    # Assess dispersion: Pearson chi2 / df_resid
    pearson_chi2 = np.sum(poisson_results.resid_pearson**2)
    dispersion = pearson_chi2 / poisson_results.df_resid if poisson_results.df_resid > 0 else np.nan

    # If overdispersion detected, fit Negative Binomial
    if not np.isnan(dispersion) and dispersion > 1.5:
        nb_mod = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
        nb_results = nb_mod.fit()
        return {
            'chosen_model': 'NegativeBinomial',
            'poisson_results': poisson_results,
            'nb_results': nb_results,
            'dispersion': dispersion
        }
    else:
        return {
            'chosen_model': 'Poisson',
            'poisson_results': poisson_results,
            'dispersion': dispersion
        }


