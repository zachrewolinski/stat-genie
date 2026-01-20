from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/fish/noperturb_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Required columns: fish_caught, hours; drop rows with missing essential values
    df = df.dropna(subset=['fish_caught', 'hours', 'persons', 'child', 'livebait', 'camper'])

    # Remove rows with nonpositive hours (cannot be used as exposure). If tiny positive values exist, keep them.
    df = df[df['hours'] > 0]

    # Ensure binary columns are integers 0/1
    df['livebait'] = df['livebait'].astype(int)
    df['camper'] = df['camper'].astype(int)

    # Create group size (adults + children)
    df['group_size'] = df['persons'] + df['child']

    # Create indicator for presence of children
    df['children_present'] = (df['child'] > 0).astype(int)

    # Create per-hour descriptive rate (not used directly in GLM but useful for exploration)
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Log of hours to be used as offset in the GLM; keep original hours too
    df['log_hours'] = np.log(df['hours'])

    # Optional: clip extreme values if necessary (not performed here). Return the transformed dataframe
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    # Build design matrix
    # Predictors: livebait, camper, group_size, children_present
    predictors = ['livebait', 'camper', 'group_size', 'children_present']
    X = df[predictors].copy()
    X = sm.add_constant(X, has_constant='add')

    # Response and offset (exposure)
    y = df['fish_caught']
    offset = df['log_hours']

    # Fit Poisson GLM with log link and offset = log(hours) to model fish count as a rate per hour
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    poisson_result = poisson_model.fit()

    # Check for overdispersion: ratio of deviance to degrees of freedom
    deviance = poisson_result.deviance
    df_resid = poisson_result.df_resid
    dispersion = deviance / df_resid if df_resid > 0 else np.nan

    # If substantial overdispersion (rule-of-thumb > 1.5), refit a Negative Binomial
    if not np.isnan(dispersion) and dispersion > 1.5:
        nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
        nb_result = nb_model.fit()
        results = nb_result
        results.model_type = 'NegativeBinomial'
        results.dispersion = dispersion
    else:
        results = poisson_result
        results.model_type = 'Poisson'
        results.dispersion = dispersion

    # Attach commonly useful summaries to results for downstream use
    results.predictors = predictors
    results.offset_col = 'log_hours'

    return results


