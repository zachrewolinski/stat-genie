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
    """
    Transform the raw fishing trips dataframe to a modeling-ready dataframe.

    Produces the following derived columns used in the model:
    - fish_per_hour: fish_caught / hours (continuous, descriptive)
    - log_hours: natural log of hours (used as offset/exposure in count model)

    Ensures numeric types, filters invalid rows (missing counts or nonpositive hours), and coerces binary indicators to ints.
    """
    df = df.copy()

    # Ensure numeric columns exist and convert where possible
    numeric_cols = ['fish_caught', 'hours', 'persons', 'camper', 'livebait', 'child']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the primary outcome or hours (exposure)
    df = df.dropna(subset=['fish_caught', 'hours'])

    # Remove non-positive hours (can't compute rate/exposure)
    df = df[df['hours'] > 0]

    # Create fish per hour rate for descriptive summaries
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Create log_hours for use as an offset in GLM
    df['log_hours'] = np.log(df['hours'])

    # Clean binary predictors: coerce to 0/1 ints where appropriate
    if 'livebait' in df.columns:
        df['livebait'] = df['livebait'].fillna(0).astype(int)
    if 'child' in df.columns:
        df['child'] = df['child'].fillna(0).astype(int)

    # Fill missing group-size / camper with sensible defaults if present
    if 'persons' in df.columns:
        # If persons missing, default to 1 to avoid dropping too many rows; user may change policy
        df['persons'] = df['persons'].fillna(1).astype(int)
    if 'camper' in df.columns:
        df['camper'] = df['camper'].fillna(0).astype(int)

    # Keep only columns required for analysis plus originals for transparency
    required_columns = [c for c in ['fish_caught', 'hours', 'fish_per_hour', 'log_hours', 'livebait', 'persons', 'camper', 'child'] if c in df.columns]
    df = df[required_columns].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a count regression for number of fish caught using hours as exposure (offset).

    Strategy:
    - Choose Poisson if variance is close to mean; otherwise use Negative Binomial to account for overdispersion.
    - Model formula: fish_caught ~ livebait + persons + camper + child
    - Use log_hours as an offset so coefficients represent multiplicative effects on fish-per-hour rates.

    Returns the fitted statsmodels results object (GLMResults).
    """
    df = df.copy()

    # Check required columns
    required = ['fish_caught', 'log_hours']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in dataframe passed to model().")

    # Basic overdispersion check
    mean_count = df['fish_caught'].mean()
    var_count = df['fish_caught'].var()

    # Select family: if variance substantially exceeds mean, prefer Negative Binomial
    if var_count > 1.5 * max(mean_count, 1e-8):
        family = sm.families.NegativeBinomial()
    else:
        family = sm.families.Poisson()

    # Build formula using predictors that exist in the dataframe
    predictors = [p for p in ['livebait', 'persons', 'camper', 'child'] if p in df.columns]
    if len(predictors) == 0:
        raise ValueError('No predictors found in dataframe for the model.')

    formula = 'fish_caught ~ ' + ' + '.join(predictors)

    # Fit GLM with offset (log of hours) so model estimates rates per hour
    model = sm.GLM.from_formula(formula, data=df, family=family, offset=df['log_hours'])
    results = model.fit()

    # Return the fitted results object. Users can call results.summary() or inspect params/conf_int()
    return results


