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
    Transform the raw dataset into a dataframe ready for modeling.

    Steps:
    - Rename feature columns to descriptive names used in the model.
    - Drop rows with missing values in essential columns.
    - Ensure correct dtypes for binary flags and numeric columns.
    - Remove rows with non-positive Hours (cannot be used as exposure).
    - Derive GroupSize (Adults + Children), FishPerHour, and an interaction term LiveBait_Camper.

    Returns the transformed dataframe containing all columns referenced in the conceptual variables and modeling code.
    """
    df = df.copy()

    # Rename features to descriptive column names used in the analysis
    df = df.rename(columns={
        'feature1': 'FishCaught',
        'feature2': 'LiveBait',
        'feature3': 'Camper',
        'feature4': 'Adults',
        'feature5': 'Children',
        'feature6': 'Hours'
    })

    # Keep only rows with non-missing essential variables
    df = df.dropna(subset=['FishCaught', 'LiveBait', 'Camper', 'Adults', 'Children', 'Hours'])

    # Coerce types: binary flags to int, counts/numeric to numeric
    df['LiveBait'] = df['LiveBait'].astype(int)
    df['Camper'] = df['Camper'].astype(int)

    # Adults/Children may be floats in raw; coerce to integer counts
    df['Adults'] = pd.to_numeric(df['Adults'], errors='coerce').astype(int)
    df['Children'] = pd.to_numeric(df['Children'], errors='coerce').astype(int)

    # Hours must be positive numeric (exposure)
    df['Hours'] = pd.to_numeric(df['Hours'], errors='coerce')
    df = df[df['Hours'] > 0].copy()

    # Derived columns
    df['GroupSize'] = df['Adults'] + df['Children']
    # Rate per hour for descriptive summaries
    df['FishPerHour'] = df['FishCaught'] / df['Hours']

    # Interaction term between LiveBait and Camper (useful if combined effect suspected)
    df['LiveBait_Camper'] = df['LiveBait'] * df['Camper']

    # Final: ensure FishCaught is integer count
    df['FishCaught'] = pd.to_numeric(df['FishCaught'], errors='coerce').astype(int)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count regression model to estimate factors influencing catch rate (fish per hour).

    Approach:
    - Use Poisson GLM with a log link and an offset = log(Hours) so the model estimates rate per hour.
    - Covariates: LiveBait, Camper, Adults, Children, and their LiveBait*Camper interaction.
    - Check for overdispersion (deviance / df_resid). If substantially > 1 (use threshold 1.5), fit a Negative Binomial GLM as an alternative.

    Returns a dictionary with the chosen model type, fitted results object, and an overdispersion metric.
    """
    df = df.copy()

    # Prepare design matrix
    covariates = ['LiveBait', 'Camper', 'Adults', 'Children', 'LiveBait_Camper']
    X = df[covariates].copy()
    X = sm.add_constant(X)
    y = df['FishCaught']

    # Offset for exposure (hours) - use log(Hours)
    offset = np.log(df['Hours'].astype(float))

    # Fit Poisson GLM
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    poisson_res = poisson_model.fit()

    # Assess overdispersion
    deviance = float(poisson_res.deviance)
    df_resid = float(poisson_res.df_resid) if hasattr(poisson_res, 'df_resid') else np.nan
    overdispersion = deviance / df_resid if df_resid and not np.isnan(df_resid) and df_resid > 0 else np.nan

    # If overdispersed, fit Negative Binomial (GLM) as alternative
    if not np.isnan(overdispersion) and overdispersion > 1.5:
        # Use a NegativeBinomial family; alpha parameter can be tuned but we use default/1.0 as starting point
        try:
            nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
            nb_res = nb_model.fit()
            return {
                'model_type': 'NegativeBinomial_GLM',
                'results': nb_res,
                'overdispersion': overdispersion
            }
        except Exception:
            # Fall back to returning Poisson results if NB fails
            return {
                'model_type': 'Poisson_GLM_fallback_due_to_NB_error',
                'results': poisson_res,
                'overdispersion': overdispersion
            }
    else:
        return {
            'model_type': 'Poisson_GLM',
            'results': poisson_res,
            'overdispersion': overdispersion
        }


