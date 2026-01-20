from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/hurricane/noperturb_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and derive variables needed for modeling the effect of hurricane name femininity on outcomes.

    Returned dataframe contains (at minimum) the columns referenced in the conceptual variables and the model:
      - masfem_z: standardized masfem index
      - alldeaths: integer count of deaths
      - gender_mf: binary 0/1 indicator
      - wind, category, min, elapsedyrs, year, source
      - ndam15 (original damage), log_ndam15 (log-transformed damage) for robustness

    The function drops rows with missing values in key columns used by the models.
    """
    df = df.copy()

    # Required original columns for analyses
    required_cols = ['masfem', 'gender_mf', 'alldeaths', 'ndam15', 'wind', 'category', 'min', 'elapsedyrs', 'year', 'source']

    # Drop rows that are missing any of the required columns
    df = df.dropna(subset=required_cols)

    # Standardize the continuous masfem score (mean 0, SD 1)
    # Use population std (ddof=0) for stability; this is arbitrary but consistent with many analyses
    df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / df['masfem'].std(ddof=0)

    # Ensure binary gender_mf is numeric 0/1
    df['gender_mf'] = pd.to_numeric(df['gender_mf'], errors='coerce')

    # Create log-transformed damage as an alternative dependent variable (add 1 to handle zeros)
    df['log_ndam15'] = np.log(df['ndam15'] + 1)

    # Ensure alldeaths is integer and non-negative
    # Coerce to integer safely (if there are floats representing counts)
    df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce').fillna(0).astype(int)

    # Ensure categorical variables are typed appropriately
    df['source'] = df['source'].astype('category')

    # Final drop for any NA introduced by coercion
    final_needed = ['masfem_z', 'gender_mf', 'alldeaths', 'log_ndam15', 'wind', 'category', 'min', 'elapsedyrs', 'year', 'source']
    df = df.dropna(subset=final_needed)

    # Reset index for a clean dataframe
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit two complementary models to test whether more feminine hurricane names are associated with fewer precautions (proxied by higher fatalities and higher damages):
      1) Negative binomial regression for death counts (alldeaths) to account for count data and overdispersion.
      2) OLS regression on log-transformed damage (log_ndam15) as a continuous robustness check.

    Returns a dict with the fitted result objects so the caller can inspect summaries, coefficients, and diagnostics.
    """
    import statsmodels.api as _sm
    import statsmodels.formula.api as smf

    # Model formula: primary predictors + controls
    formula = (
        'alldeaths ~ masfem_z + gender_mf + wind + category + min + elapsedyrs + C(source) + year'
    )

    # Fit a Negative Binomial GLM for count outcome (alldeaths)
    # Using statsmodels' GLM with NegativeBinomial family
    nb_model = smf.glm(formula=formula, data=df, family=_sm.families.NegativeBinomial()).fit()

    # Complementary model: OLS on log-transformed damage (robust se)
    formula_damage = (
        'log_ndam15 ~ masfem_z + gender_mf + wind + category + min + elapsedyrs + C(source) + year'
    )
    ols_damage = smf.ols(formula=formula_damage, data=df).fit(cov_type='HC3')

    # Return the fitted results so the caller can inspect coefficients and summaries
    return {
        'neg_binom_alldeaths': nb_model,
        'ols_log_ndam15': ols_damage
    }


