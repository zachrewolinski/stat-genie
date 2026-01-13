from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/fish/shuffle_names_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure required columns exist and are numeric
    required = ['fish_caught', 'hours', 'livebait', 'persons', 'camper', 'child']
    # coerce columns to numeric when possible
    for c in required:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the essential variables
    df = df.dropna(subset=['fish_caught', 'hours'])

    # Remove or floor negative/implausible values
    # fish_caught should be >= 0, hours should be >= 0.25 (practical lower bound)
    df = df[df['fish_caught'] >= 0]
    df = df[df['hours'] > 0]

    # Create a rate: fish caught per hour
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Create an offset for GLM (log of exposure hours)
    df['log_hours'] = np.log(df['hours'])

    # Ensure binary columns are 0/1 integers
    if 'child' in df.columns:
        df['child'] = df['child'].astype(float).fillna(0).astype(int)
    if 'livebait' in df.columns:
        df['livebait'] = df['livebait'].astype(float).fillna(0).astype(int)

    # Ensure integer counts where expected
    if 'persons' in df.columns:
        # persons might be reported as non-integer; coerce to integer if appropriate
        df['persons'] = pd.to_numeric(df['persons'], errors='coerce').fillna(0).astype(int)
    if 'camper' in df.columns:
        df['camper'] = pd.to_numeric(df['camper'], errors='coerce').fillna(0).astype(int)

    # (Optional) Filter extreme outliers in fish_per_hour that likely reflect data entry errors
    # For robustness, remove top 0.5% of fish_per_hour
    if 'fish_per_hour' in df.columns:
        upper = df['fish_per_hour'].quantile(0.995)
        df = df[df['fish_per_hour'] <= upper]

    # Finalize: keep only columns necessary for modeling
    model_cols = ['fish_caught', 'fish_per_hour', 'hours', 'log_hours', 'livebait', 'persons', 'camper', 'child']
    keep = [c for c in model_cols if c in df.columns]
    return df[keep].copy()


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a count-style model for fish caught with hours as exposure.
    Primary approach: Poisson GLM with log(hours) as an offset. If overdispersion is detected,
    refit with a Negative Binomial GLM.

    Returns a dict with the Poisson fit, the chosen final model (Poisson or NB), and a dispersion statistic.
    """
    # Exogenous variables (covariates)
    exog_cols = ['livebait', 'persons', 'camper', 'child']
    for c in exog_cols:
        if c not in df.columns:
            raise ValueError(f"Required covariate missing from dataframe: {c}")

    X = df[exog_cols].copy()
    X = sm.add_constant(X, has_constant='add')

    # Response is the raw count 'fish_caught'; use log(hours) as offset (exposure)
    y = df['fish_caught']
    offset = df['log_hours']

    # Fit Poisson GLM
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    poisson_results = poisson_model.fit()

    # Check for overdispersion: Pearson chi2 / df_resid or deviance/df_resid
    # Use deviance-based dispersion estimate
    try:
        dispersion = poisson_results.deviance / poisson_results.df_resid
    except Exception:
        # fallback to Pearson chi2
        pearson_chi2 = ((poisson_results.resid_pearson) ** 2).sum()
        dispersion = pearson_chi2 / poisson_results.df_resid

    final_results = poisson_results
    final_family = 'Poisson'

    # If substantial overdispersion, switch to Negative Binomial
    if dispersion > 1.5:
        nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
        try:
            nb_results = nb_model.fit()
            final_results = nb_results
            final_family = 'NegativeBinomial'
        except Exception:
            # If NB fails to converge, keep Poisson but report dispersion
            final_results = poisson_results
            final_family = 'Poisson (NB failed)'

    # Return results objects and diagnostic numbers. The caller can inspect .summary() on returned results.
    return {
        'poisson_results': poisson_results,
        'final_model': final_results,
        'final_family': final_family,
        'dispersion': dispersion,
        'exog_columns': X.columns.tolist()
    }


