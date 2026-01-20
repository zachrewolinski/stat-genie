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
    Transform raw fishing trip data to a dataframe ready for count-rate modeling.
    Produces columns used by the model: fish_caught, livebait, camper, group_size, hours, log_hours, fish_per_hour.

    Steps:
    - Drop rows with missing critical fields (fish_caught, hours, livebait, camper, persons, child).
    - Drop or exclude rows with non-positive hours (cannot model rate / offset log(0)).
    - Create group_size = persons + child.
    - Create fish_per_hour for descriptive diagnostics.
    - Create log_hours = log(hours) for use as offset in GLM.
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Required columns - ensure they exist
    required_cols = ['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in dataframe: {missing}")

    # Drop rows with missing values in critical columns
    df = df.dropna(subset=required_cols)

    # Ensure numeric types
    df['fish_caught'] = pd.to_numeric(df['fish_caught'], errors='coerce')
    df['hours'] = pd.to_numeric(df['hours'], errors='coerce')
    df['livebait'] = pd.to_numeric(df['livebait'], errors='coerce')
    df['camper'] = pd.to_numeric(df['camper'], errors='coerce')
    df['persons'] = pd.to_numeric(df['persons'], errors='coerce')
    df['child'] = pd.to_numeric(df['child'], errors='coerce')

    # Drop rows introduced as NaN by conversion
    df = df.dropna(subset=required_cols)

    # Exclude rows with non-positive hours (cannot take log); if very small positive hours are present keep them
    df = df[df['hours'] > 0]

    # Create group size (total people present)
    df['group_size'] = df['persons'] + df['child']

    # Create descriptive rate column (fish per hour)
    # Avoid division issues - hours > 0 ensured above
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Create log_hours for offset in GLM
    df['log_hours'] = np.log(df['hours'])

    # Ensure binary columns are 0/1 integers
    df['livebait'] = df['livebait'].astype(int)
    df['camper'] = df['camper'].astype(int)

    # Optional: remove extreme outliers for group_size (if unrealistic) - keep as-is by default

    # Final columns to keep (order for readability); model uses subset further
    final_cols = ['fish_caught', 'livebait', 'camper', 'persons', 'child', 'group_size', 'hours', 'log_hours', 'fish_per_hour']
    # Some columns might not be present if original lacked persons/child - they were checked above
    df = df[final_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count regression to estimate rate of fish caught per hour and the effects of predictors.

    Approach:
    - Use fish_caught as the count response and include log_hours as an offset to model rate (fish per hour).
    - Predictor set: livebait (primary IV), camper, group_size (controls).
    - Check for overdispersion (variance / mean). If overdispersion is moderate-to-high (dispersion > 1.5), fit a Negative Binomial (GLM) to account for extra-Poisson variability. Otherwise fit a Poisson GLM.

    Returns the fitted results object (statsmodels regression results). Also prints a short summary and dispersion diagnostics.
    """
    # Ensure required columns exist
    required = ['fish_caught', 'livebait', 'camper', 'group_size', 'log_hours']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for modeling: {missing}")

    # Compute dispersion diagnostic
    mean_count = df['fish_caught'].mean()
    var_count = df['fish_caught'].var()
    dispersion = var_count / mean_count if mean_count > 0 else np.nan

    print(f"Mean fish caught: {mean_count:.3f}, Variance: {var_count:.3f}, Dispersion (var/mean): {dispersion:.3f}")

    # Design matrix
    X = df[['livebait', 'camper', 'group_size']].copy()
    X = sm.add_constant(X)
    y = df['fish_caught']
    offset = df['log_hours']

    # Choose model family based on dispersion
    if np.isfinite(dispersion) and dispersion > 1.5:
        print("Overdispersion detected (dispersion > 1.5). Fitting Negative Binomial GLM with log offset.")
        family = sm.families.NegativeBinomial()
    else:
        print("No strong overdispersion detected. Fitting Poisson GLM with log offset.")
        family = sm.families.Poisson()

    model = sm.GLM(y, X, family=family, offset=offset)
    results = model.fit()

    # Print summary (user can inspect results returned)
    print(results.summary())

    # For convenience also return a small dictionary with key model objects
    return {
        'results': results,
        'model': model,
        'dispersion': dispersion,
        'mean_count': mean_count,
        'variance_count': var_count
    }


