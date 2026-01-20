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
    Clean and derive variables needed for modeling fishing rate.

    Produces:
      - group_size: persons + child
      - fish_per_hour: fish_caught / hours (descriptive)
      - log_hours: natural log of hours (offset for count model)
      - casts of binary vars to int and removal of rows with invalid hours or missing key data

    Returns a dataframe containing all columns referred to in the conceptual variables and used in the model.
    """
    df = df.copy()

    # Required raw columns for modeling
    required_cols = ['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child']
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError('Missing required columns: ' + ','.join(missing))

    # Drop rows missing essential numeric values
    df = df.dropna(subset=['fish_caught', 'hours'])

    # Ensure numeric types for key columns
    df['fish_caught'] = pd.to_numeric(df['fish_caught'], errors='coerce')
    df['hours'] = pd.to_numeric(df['hours'], errors='coerce')
    df['persons'] = pd.to_numeric(df['persons'], errors='coerce').fillna(0).astype(int)
    df['child'] = pd.to_numeric(df['child'], errors='coerce').fillna(0).astype(int)

    # Remove rows with nonpositive hours (cannot compute rate / offset)
    df = df[df['hours'] > 0].copy()

    # Ensure binary indicators are integers (0/1)
    if 'livebait' in df.columns:
        df['livebait'] = pd.to_numeric(df['livebait'], errors='coerce').fillna(0).astype(int)
    if 'camper' in df.columns:
        df['camper'] = pd.to_numeric(df['camper'], errors='coerce').fillna(0).astype(int)

    # Derived variables
    df['group_size'] = df['persons'] + df['child']
    # Descriptive per-hour rate (not used as DV in the GLM) - keep for summaries and diagnostics
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Offset for Poisson/Negative Binomial modeling
    df['log_hours'] = np.log(df['hours'])

    # Keep a set of columns expected by the model / analysis. If some optional columns aren't present, return those that are.
    desired = [
        'fish_caught', 'hours', 'log_hours', 'fish_per_hour',
        'livebait', 'camper', 'persons', 'child', 'group_size',
        'religiousness', 'age', 'county'
    ]
    present = [c for c in desired if c in df.columns]

    return df[present]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a count model for fish_caught using an exposure offset (log_hours) so coefficients represent multiplicative effects on the rate (fish per hour).

    Model specification (default): Negative Binomial GLM with predictors:
      - livebait, camper, group_size, religiousness, age
    Offset: log_hours (so outcome is fish count and offset controls for hours spent)

    Returns a dictionary with:
      - 'model': fitted statsmodels results object
      - 'predicted_count': predicted expected fish counts for each row
      - 'predicted_rate_per_hour': predicted expected fish per hour for each row
      - 'avg_predicted_rate_per_hour': average predicted fish per hour across dataset
    """
    # Copy dataframe to avoid side-effects
    df = df.copy()

    # Choose predictors (only include predictors that are present in df)
    candidate_preds = ['livebait', 'camper', 'group_size', 'religiousness', 'age']
    preds = [p for p in candidate_preds if p in df.columns]
    if len(preds) == 0:
        raise ValueError('No predictors available in dataframe for modeling.')

    formula = 'fish_caught ~ ' + ' + '.join(preds)

    # Use Negative Binomial to allow for overdispersion relative to Poisson
    # Use offset = log_hours if present
    if 'log_hours' in df.columns:
        fam = sm.families.NegativeBinomial()
        model_fit = sm.GLM.from_formula(formula, data=df, family=fam, offset=df['log_hours']).fit()
    else:
        # Fallback: fit without offset (not recommended for rate inference)
        fam = sm.families.NegativeBinomial()
        model_fit = sm.GLM.from_formula(formula, data=df, family=fam).fit()

    # Predicted expected counts (on same scale as fish_caught)
    predicted_count = model_fit.predict(df)

    # Convert to predicted rate per hour if hours column exists
    if 'hours' in df.columns:
        predicted_rate_per_hour = predicted_count / df['hours']
        avg_predicted_rate_per_hour = float(predicted_rate_per_hour.mean())
    else:
        predicted_rate_per_hour = None
        avg_predicted_rate_per_hour = None

    results = {
        'model': model_fit,
        'predicted_count': predicted_count,
        'predicted_rate_per_hour': predicted_rate_per_hour,
        'avg_predicted_rate_per_hour': avg_predicted_rate_per_hour
    }

    return results


