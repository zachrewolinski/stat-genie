from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/fish/noperturb_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw park fishing dataset to the analysis-ready dataframe.

    Output columns (in addition to original):
      - total_people: persons + child
      - total_people_c: mean-centered total_people (used as predictor)
      - fish_per_hour: descriptive rate fish_caught / hours
      - log_hours: natural log of hours (used as offset in GLM)

    Drops rows with missing critical values and rows with hours <= 0.
    """
    # Work on a copy
    df = df.copy()

    # Drop rows with missing values in the core columns needed for the analysis
    required_cols = ['fish_caught', 'livebait', 'camper', 'persons', 'child', 'hours']
    df = df.dropna(subset=required_cols)

    # Remove rows with non-positive or extremely small hours (cannot take log of 0 or negative)
    # Keep a tolerant threshold for floating noise: hours must be > 0.0
    df = df[df['hours'] > 0.0]

    # Create total group size and mean-center it for modeling stability
    df['total_people'] = df['persons'].astype(float) + df['child'].astype(float)
    df['total_people_c'] = df['total_people'] - df['total_people'].mean()

    # Descriptive per-hour rate (useful for summaries/plots)
    df['fish_per_hour'] = df['fish_caught'].astype(float) / df['hours'].astype(float)

    # Offset (log of hours) for use in Poisson/NegBin GLM
    df['log_hours'] = np.log(df['hours'].astype(float))

    # Ensure binary columns are ints (0/1)
    df['livebait'] = df['livebait'].astype(int)
    df['camper'] = df['camper'].astype(int)

    # Final dataframe returned should include the original relevant columns plus derived ones
    # (No further filtering here; model code can decide if further filtering is needed.)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit count regression models to estimate factors that influence fish caught per hour.

    Strategy:
    1) Fit a Poisson GLM with offset = log_hours.
    2) Compute dispersion (Pearson chi-square / df_resid) to assess overdispersion.
    3) Fit a Negative Binomial GLM as a robust alternative if overdispersion is present.

    Predictors: livebait, camper, total_people_c. Dependent variable: fish_caught. Offset: log_hours.

    Returns a dictionary with Poisson and Negative Binomial fitted results and diagnostic info.
    """
    results = {}

    # Ensure required columns exist
    for col in ['fish_caught', 'livebait', 'camper', 'total_people_c', 'log_hours']:
        if col not in df.columns:
            raise ValueError(f"Required column missing from dataframe: {col}")

    # Define endogenous and exogenous variables
    endog = df['fish_caught'].astype(float)
    exog = df[['livebait', 'camper', 'total_people_c']].astype(float)
    exog = sm.add_constant(exog, has_constant='add')
    offset = df['log_hours'].astype(float)

    # Fit Poisson GLM with offset
    poisson_model = sm.GLM(endog, exog, family=sm.families.Poisson(), offset=offset)
    poisson_res = poisson_model.fit()

    # Compute dispersion (Pearson chi2 / df_resid) for Poisson
    pearson_chi2 = sum(poisson_res.resid_pearson ** 2)
    df_resid = poisson_res.df_resid if hasattr(poisson_res, 'df_resid') else (len(df) - exog.shape[1])
    dispersion = pearson_chi2 / float(df_resid) if df_resid > 0 else np.nan

    results['poisson'] = poisson_res
    results['dispersion'] = dispersion

    # Fit Negative Binomial GLM (more flexible for overdispersion)
    # Always fit NB for comparison; if dispersion approx 1 Poisson is fine, if >1 NB usually preferable.
    try:
        nb_model = sm.GLM(endog, exog, family=sm.families.NegativeBinomial(), offset=offset)
        nb_res = nb_model.fit()
        results['neg_binom'] = nb_res
    except Exception as e:
        # If NegativeBinomial GLM fails (rare), include the exception message
        results['neg_binom'] = None
        results['neg_binom_error'] = str(e)

    # Add simple descriptive statistics for interpretation
    results['descriptives'] = {
        'n_rows': int(len(df)),
        'mean_fish_caught': float(df['fish_caught'].mean()),
        'var_fish_caught': float(df['fish_caught'].var()),
        'mean_fish_per_hour': float(df['fish_per_hour'].mean())
    }

    # Return the fitted objects and diagnostics. The caller can inspect .summary() on the results.
    return results


