from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/fish/add_features_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the input dataframe to prepare variables for modeling fish caught per hour.

    Produces the following new/ensured columns used in the statistical model:
    - total_people: persons + child
    - fish_per_hour: fish_caught / hours (descriptive)
    - ensures livebait and camper are integer 0/1
    - removes rows with missing or non-positive hours
    """
    # Work on a copy
    df = df.copy()

    # Drop rows missing the core outcome or exposure
    df = df.dropna(subset=['fish_caught', 'hours'])

    # Remove rows with non-positive hours (can't take log for offset)
    df = df[df['hours'] > 0]

    # Ensure binary predictors are integer 0/1 (if there are missing values, fill with 0)
    for col in ['livebait', 'camper']:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Ensure persons and child exist; fill reasonable defaults if missing
    if 'persons' in df.columns:
        df['persons'] = df['persons'].fillna(1)
    else:
        # if persons not present, create as 1 (single adult) to avoid errors
        df['persons'] = 1

    if 'child' in df.columns:
        df['child'] = df['child'].fillna(0)
    else:
        df['child'] = 0

    # Create total_people control variable
    df['total_people'] = df['persons'] + df['child']

    # Descriptive rate column (not used directly in the GLM; GLM uses fish_caught + offset)
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # If county exists, keep as-is (categorical). Fill missing county with 'Unknown' to allow C(county) in model
    if 'county' in df.columns:
        df['county'] = df['county'].fillna('Unknown')

    # Final dataframe returned contains at minimum the columns used in the model
    # (fish_caught, hours, livebait, camper, total_people, county (if present)).
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Fit a rate model for fish caught per hour.

    Steps:
    1. Fit a Poisson GLM with log(hours) as an offset to model fish_caught as a rate per hour.
    2. Compute a dispersion (overdispersion) metric using the Pearson chi-square / residual df.
    3. If substantial overdispersion observed (dispersion > 1.5), fit a Negative Binomial GLM as a robustness/alternative.

    Returns a dictionary with keys:
    - 'poisson_model': fitted Poisson results (statsmodels object)
    - 'overdispersion': computed dispersion metric
    - 'negbin_model': fitted Negative Binomial results (if fit, otherwise omitted)
    - 'formula': formula used
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import numpy as np

    # Work on a copy
    df = df.copy()

    # Build offset (log of hours). hours should be > 0 due to transform.
    df['log_hours'] = np.log(df['hours'])

    # Base formula: main predictors + control for group size. Add county fixed effects if present.
    formula = 'fish_caught ~ livebait + camper + total_people'
    if 'county' in df.columns:
        formula += ' + C(county)'

    # Fit Poisson GLM with offset
    poisson_model = smf.glm(formula=formula, data=df, family=sm.families.Poisson(), offset=df['log_hours']).fit()

    # Compute Pearson chi-square dispersion measure
    y = df['fish_caught'].values
    mu = poisson_model.fittedvalues.values
    # Avoid division by zero: where mu is zero, set small value
    mu_safe = np.where(mu <= 0, 1e-8, mu)
    pearson_chi2 = np.sum(((y - mu_safe) ** 2) / mu_safe)

    # residual degrees of freedom: n - p (poisson_model.df_model includes number of regressors excluding intercept)
    n_obs = df.shape[0]
    # df_model is number of regressors (not including intercept) for statsmodels GLM results; add 1 for intercept
    p = int(poisson_model.df_model) + 1
    rdf = n_obs - p
    dispersion = pearson_chi2 / rdf if rdf > 0 else np.nan

    results = {
        'poisson_model': poisson_model,
        'overdispersion': dispersion,
        'formula': formula
    }

    # If overdispersed, fit Negative Binomial GLM as an alternative
    if (not np.isnan(dispersion)) and (dispersion > 1.5):
        negbin_model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=df['log_hours']).fit()
        results['negbin_model'] = negbin_model

    # Also include a small summary of key fit quantities for easy inspection
    results['poisson_aic'] = poisson_model.aic
    if 'negbin_model' in results:
        results['negbin_aic'] = results['negbin_model'].aic

    return results


