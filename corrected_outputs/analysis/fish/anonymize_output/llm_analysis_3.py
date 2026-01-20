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
    Transform the original dataset into a cleaned dataframe ready for count regression.

    Produces the following columns used by the model:
      - FishCaught: integer count of fish caught (from feature1)
      - LiveBait: 0/1 indicator (from feature2)
      - Camper: 0/1 indicator (from feature3)
      - Adults: integer number of adults (from feature4)
      - Children: integer number of children (from feature5)
      - Hours: float number of hours spent in park (from feature6)
      - GroupSize: Adults + Children (derived, for diagnostics)
      - Rate_per_hour: FishCaught / Hours (derived, descriptive)
      - log_Hours: natural log of Hours (for use as an offset in GLM)
    """
    df = df.copy()

    # Rename features to descriptive column names used in modeling
    rename_map = {
        'feature1': 'FishCaught',
        'feature2': 'LiveBait',
        'feature3': 'Camper',
        'feature4': 'Adults',
        'feature5': 'Children',
        'feature6': 'Hours'
    }
    df = df.rename(columns=rename_map)

    # Convert numeric columns, coerce errors to NaN
    to_numeric_cols = ['FishCaught', 'LiveBait', 'Camper', 'Adults', 'Children', 'Hours']
    for c in to_numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with missing required values
    df = df.dropna(subset=['FishCaught', 'LiveBait', 'Camper', 'Adults', 'Children', 'Hours'])

    # Remove implausible / invalid rows: negative fish or nonpositive hours
    df = df[df['FishCaught'] >= 0]
    df = df[df['Hours'] > 0]

    # Cast integer-like columns to integer type where appropriate
    df['FishCaught'] = df['FishCaught'].astype(int)
    df['LiveBait'] = df['LiveBait'].astype(int)
    df['Camper'] = df['Camper'].astype(int)
    df['Adults'] = df['Adults'].astype(int)
    df['Children'] = df['Children'].astype(int)

    # Derived variables
    df['GroupSize'] = df['Adults'] + df['Children']
    # Rate for descriptive analysis
    df['Rate_per_hour'] = df['FishCaught'] / df['Hours']
    # log of hours for offset (GLM expects numeric offset)
    df['log_Hours'] = np.log(df['Hours'])

    # Keep only the columns that matter for modeling + useful diagnostics
    keep_cols = ['FishCaught', 'LiveBait', 'Camper', 'Adults', 'Children', 'Hours', 'GroupSize', 'Rate_per_hour', 'log_Hours']
    df = df[keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a count regression to estimate factors associated with number of fish caught per hour.

    Steps:
    1. Fit a Poisson GLM with log(Hours) as an offset to model fish caught with exposure = hours (i.e., rate per hour).
    2. Compute dispersion (Pearson chi-square / df_resid). If there is evidence of overdispersion (dispersion > 1.5), fit a Negative Binomial GLM instead.
    3. Return the fitted models and diagnostics; indicate which model was chosen.

    Model formula:
      FishCaught ~ LiveBait + Camper + Adults + Children
      offset = log_Hours
    """
    import statsmodels.formula.api as smf

    # Ensure a copy so we don't modify original
    df = df.copy()

    # Basic sanity check
    required = ['FishCaught', 'LiveBait', 'Camper', 'Adults', 'Children', 'log_Hours', 'Hours']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataframe is missing required columns for modeling: {missing}")

    # Formula (response is count). Hours provided as offset via log_Hours column
    formula = 'FishCaught ~ LiveBait + Camper + Adults + Children'

    # Fit Poisson GLM with offset
    poisson_model = smf.glm(formula=formula,
                            data=df,
                            family=sm.families.Poisson(),
                            offset=df['log_Hours']).fit()

    # Calculate dispersion using Pearson residuals
    pearson_chi2 = (poisson_model.resid_pearson ** 2).sum()
    df_resid = poisson_model.df_resid if poisson_model.df_resid is not None else max(1, df.shape[0] - poisson_model.df_model - 1)
    dispersion = pearson_chi2 / df_resid

    results = {
        'poisson_model': poisson_model,
        'dispersion': dispersion,
        'chosen_model': 'poisson'  # may update below
    }

    # If overdispersion is present, fit Negative Binomial GLM
    # Threshold of 1.5 is a practical rule-of-thumb; you can change it based on domain needs
    if dispersion > 1.5:
        try:
            nb_model = smf.glm(formula=formula,
                               data=df,
                               family=sm.families.NegativeBinomial(),
                               offset=df['log_Hours']).fit()
            results['negative_binomial_model'] = nb_model
            results['chosen_model'] = 'negative_binomial'
        except Exception as e:
            # If NB fitting fails, keep Poisson and note the error
            results['negative_binomial_error'] = str(e)
            results['chosen_model'] = 'poisson'

    return results


