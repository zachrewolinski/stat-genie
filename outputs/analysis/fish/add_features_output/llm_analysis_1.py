from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/fish/add_features_output/fish.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Required columns for analysis
    required_cols = ['fish_caught', 'hours', 'livebait', 'camper', 'persons', 'child']

    # Drop rows missing core variables
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    # Remove non-positive hours (can't compute a rate/exposure)
    if 'hours' in df.columns:
        df = df[df['hours'] > 0].copy()

    # Derive group_size (adults + children)
    if ('persons' in df.columns) and ('child' in df.columns):
        df['group_size'] = df['persons'] + df['child']
    else:
        # if persons or child missing, try to create group_size from available columns
        if 'persons' in df.columns:
            df['group_size'] = df['persons']
        elif 'child' in df.columns:
            df['group_size'] = df['child']

    # Derive fish per hour for descriptive summaries
    df['fish_per_hour'] = df['fish_caught'] / df['hours']

    # Create log-hours offset for count model (exposure)
    df['offset'] = np.log(df['hours'])

    # Clean county text (trim whitespace) and preserve missing as NaN
    if 'county' in df.columns:
        df['county'] = df['county'].astype(str).str.strip()
        df.loc[df['county'].isin(['nan', 'None', '']), 'county'] = np.nan

    # Ensure binary indicators are numeric 0/1
    for col in ['livebait', 'camper']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure numeric columns are numeric
    for col in ['fish_caught', 'hours', 'group_size', 'year', 'age']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows that lost required numeric values after coercion
    keep_cols = ['fish_caught', 'offset', 'livebait', 'camper', 'group_size']
    df = df.dropna(subset=[c for c in keep_cols if c in df.columns])

    # Final: reset index for downstream modeling convenience
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    df = df.copy()

    # Select predictors we will include if present
    base_predictors = [p for p in ['livebait', 'camper', 'group_size', 'year', 'age'] if p in df.columns]

    # Build county dummies if county is available
    if 'county' in df.columns:
        county_dummies = pd.get_dummies(df['county'], prefix='county', drop_first=True)
        X = pd.concat([df[base_predictors].reset_index(drop=True), county_dummies.reset_index(drop=True)], axis=1)
    else:
        X = df[base_predictors].copy()

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Response and offset
    y = df['fish_caught']
    offset = df['offset'] if 'offset' in df.columns else np.log(df['hours'])

    # Fit a negative binomial GLM with log link and exposure offset to model counts with differing hours
    # This models E[fish_caught] = exp(X beta + log(hours)) => rate per hour modeled by predictors
    try:
        model_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
        results = model_nb.fit()
    except Exception:
        # Fallback to Poisson if NegativeBinomial fails to converge
        model_pois = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
        results = model_pois.fit()

    # Return the fitted results object (has summary(), params, conf_int(), etc.)
    return results


