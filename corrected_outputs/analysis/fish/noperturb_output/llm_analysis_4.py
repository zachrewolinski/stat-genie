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
    """
    Transform the raw dataset into the final dataframe used by the model.

    Produces the following additional columns (kept in the returned df):
    - total_people: persons + child
    - prop_children: child / total_people (proportion of group who are children)
    - fish_per_hour: fish_caught / hours (for descriptive summaries)
    - log_hours: natural log of hours (used as offset/exposure in count models)

    Also drops rows with missing or invalid key values (fish_caught, hours).
    """
    # copy to avoid modifying input in-place
    df = df.copy()

    # Ensure numeric columns are present and coerce if necessary
    num_cols = ['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing outcome or exposure
    df = df.dropna(subset=['fish_caught', 'hours'])

    # Remove non-positive or extremely small hours to avoid log(0)
    # Keep only rows with hours > 0
    df = df[df['hours'] > 0]

    # Fill or coerce binary indicators to 0/1 (if they are present but not integer)
    if 'livebait' in df.columns:
        df['livebait'] = df['livebait'].astype(float).fillna(0).astype(int)
    if 'camper' in df.columns:
        df['camper'] = df['camper'].astype(float).fillna(0).astype(int)

    # Create total_people and proportion of children
    df['persons'] = df['persons'].fillna(0)
    df['child'] = df['child'].fillna(0)
    df['total_people'] = df['persons'] + df['child']

    # Avoid division by zero: persons has min 1 in schema, but guard anyway
    df['total_people'] = df['total_people'].replace(0, np.nan)

    # Drop any rows that became invalid due to zero total_people
    df = df.dropna(subset=['total_people'])

    df['prop_children'] = df['child'] / df['total_people']

    # Descriptive derived column: fish per hour
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Offset/exposure for count models: log(hours)
    df['log_hours'] = np.log(df['hours'])

    # Keep only columns required for modeling and useful diagnostics
    # (but return full df with derived columns)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit count regression(s) to estimate rate of fish caught per hour.

    Approach:
    1. Fit a Poisson GLM with offset=log_hours using predictors: livebait, camper, total_people, prop_children.
    2. Compute overdispersion (Pearson chi-squared / df_resid). If substantial overdispersion (>1.5), fit a Negative Binomial GLM as an alternative.

    Returns a dictionary with keys:
    - 'poisson_model': fitted Poisson results (statsmodels object)
    - 'overdispersion': computed overdispersion statistic
    - 'negbin_model' (optional): fitted Negative Binomial results if overdispersion suggests
    - 'errors' (optional): any errors encountered during fitting
    """
    results = {}

    # Ensure required columns exist and drop rows with missing predictors/offset
    required = ['fish_caught', 'log_hours', 'livebait', 'camper', 'total_people', 'prop_children']
    df_model = df.dropna(subset=required).copy()

    # Design matrix
    X = df_model[['livebait', 'camper', 'total_people', 'prop_children']]
    X = sm.add_constant(X, has_constant='add')
    y = df_model['fish_caught']
    offset = df_model['log_hours']

    # Fit Poisson GLM with offset
    poisson_mod = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    try:
        poisson_res = poisson_mod.fit()
        results['poisson_model'] = poisson_res

        # Overdispersion: Pearson chi2 / df_resid
        pearson_chi2 = (poisson_res.resid_pearson ** 2).sum()
        overdispersion = pearson_chi2 / poisson_res.df_resid if poisson_res.df_resid > 0 else np.nan
        results['overdispersion'] = overdispersion
    except Exception as e:
        results['errors'] = {'poisson_fit_error': str(e)}
        return results

    # If overdispersion is substantial, fit Negative Binomial
    if (not np.isnan(results.get('overdispersion', np.nan))) and results['overdispersion'] > 1.5:
        try:
            negbin_mod = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
            negbin_res = negbin_mod.fit()
            results['negbin_model'] = negbin_res
        except Exception as e:
            # If GLM NegativeBinomial fails (depending on statsmodels version), attempt alternative
            results.setdefault('errors', {})['negbin_fit_error'] = str(e)

    return results


