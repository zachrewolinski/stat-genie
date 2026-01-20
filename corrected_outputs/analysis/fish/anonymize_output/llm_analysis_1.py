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
    Transform the raw dataset into a cleaned dataframe with columns used by the statistical model.

    - Renames raw columns to meaningful names
    - Drops rows with missing key variables
    - Ensures numeric types
    - Clips Hours to a small positive value if zero or negative to allow use as exposure
    - Creates FishPerHour for descriptive statistics
    - Centers Adults and Children (creates Adults_c and Children_c) for model stability/interpretation

    Returns the transformed dataframe containing at least these columns:
    ['FishCount', 'LiveBait', 'Camper', 'Adults', 'Children', 'Hours', 'FishPerHour', 'TotalPeople', 'Adults_c', 'Children_c']
    """
    df = df.copy()

    # Rename columns to meaningful names
    rename_map = {
        'feature1': 'FishCount',   # number of fish caught
        'feature2': 'LiveBait',    # 0/1
        'feature3': 'Camper',      # 0/1
        'feature4': 'Adults',      # integer
        'feature5': 'Children',    # integer
        'feature6': 'Hours'        # hours spent in park
    }
    df = df.rename(columns=rename_map)

    # Keep only relevant columns if others exist
    needed = ['FishCount', 'LiveBait', 'Camper', 'Adults', 'Children', 'Hours']
    df = df[needed].copy()

    # Drop rows with missing values in key fields
    df = df.dropna(subset=needed)

    # Convert to numeric types
    for col in ['FishCount', 'LiveBait', 'Camper', 'Adults', 'Children', 'Hours']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows that became NaN after coercion
    df = df.dropna(subset=needed)

    # Clip Hours to a small positive number if <= 0 to allow using log(Hours) as offset
    # We prefer to keep observations but avoid log(0); alternatively could drop these rows.
    df['Hours'] = df['Hours'].astype(float)
    df.loc[df['Hours'] <= 0, 'Hours'] = 0.001

    # Ensure binary predictors are integers (0/1)
    df['LiveBait'] = df['LiveBait'].astype(int)
    df['Camper'] = df['Camper'].astype(int)

    # Create FishPerHour (for descriptive checks) and TotalPeople
    df['FishPerHour'] = df['FishCount'] / df['Hours']
    df['TotalPeople'] = df['Adults'] + df['Children']

    # Center Adults and Children to improve interpretability and numerical stability
    df['Adults_c'] = df['Adults'] - df['Adults'].mean()
    df['Children_c'] = df['Children'] - df['Children'].mean()

    # Keep final columns required by the model (and some extras for diagnostics)
    final_cols = ['FishCount', 'LiveBait', 'Camper', 'Adults', 'Children', 'Hours', 'FishPerHour', 'TotalPeople', 'Adults_c', 'Children_c']
    df = df[final_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a count regression to estimate factors affecting fish caught per hour.

    Strategy:
    - Use FishCount as the dependent variable and model it as a rate with Hours as exposure (log(Hours) offset).
    - Fit a Poisson GLM first. If the count data are overdispersed (variance substantially > mean), fit a Negative Binomial GLM instead.
    - Predictors: LiveBait, Camper, Adults_c, Children_c. A constant is included.

    Returns the fitted results object (statsmodels results) and prints a brief summary and which family was used.
    """
    import numpy as np
    import statsmodels.api as sm

    df = df.copy()

    # Define predictors and response
    predictors = ['LiveBait', 'Camper', 'Adults_c', 'Children_c']
    y = df['FishCount'].astype(float)
    X = df[predictors].astype(float)
    X = sm.add_constant(X)

    # Offset is log(hours) to model counts as a rate per hour
    offset = np.log(df['Hours'].astype(float))

    # Quick dispersion check (mean/variance) to choose family
    mean_count = y.mean()
    var_count = y.var()

    # Set threshold for overdispersion heuristic
    overdispersion_threshold = 1.5  # if var > threshold * mean, treat as overdispersed

    if var_count > overdispersion_threshold * mean_count and mean_count > 0:
        family = sm.families.NegativeBinomial()
        family_name = 'NegativeBinomial'
    else:
        family = sm.families.Poisson()
        family_name = 'Poisson'

    # Fit GLM with chosen family
    try:
        model_glm = sm.GLM(y, X, family=family, offset=offset)
        results = model_glm.fit()
    except Exception as e:
        # If NB fitting via GLM fails for some reason, fallback to Poisson
        print(f"Primary fit with {family_name} failed: {e}. Falling back to Poisson.")
        family = sm.families.Poisson()
        family_name = 'Poisson'
        model_glm = sm.GLM(y, X, family=family, offset=offset)
        results = model_glm.fit()

    # Print brief diagnostics
    print(f"Selected family: {family_name}")
    print("Data mean count = {:.3f}, variance = {:.3f}".format(mean_count, var_count))
    try:
        # deviance-based dispersion for Poisson (should be ~1); for NB this is less informative
        deviance = results.deviance
        df_resid = results.df_resid
        dispersion = deviance / df_resid if df_resid > 0 else np.nan
        print(f"Deviance: {deviance:.3f}, df_resid: {df_resid}, dispersion (deviance/df_resid): {dispersion:.3f}")
    except Exception:
        pass

    print(results.summary())

    # Return results object so caller can inspect coefficients, get predicted rates, etc.
    return results


