from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/shuffle_names_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # --- Rename / select raw columns we will use ---
    # Original column meanings (from provided schema):
    #   'name'       : continuous masculinity-femininity index (higher = more feminine)
    #   'elapsedyrs' : binary gender indicator for the name (0 male, 1 female)
    #   'ndam15'     : total number of deaths caused by the hurricane (count)
    #   'wind'       : maximum wind speed at landfall
    #   'min'        : minimum central pressure at landfall
    #   'masfem'     : Saffir-Simpson category (1-5)
    #   'alldeaths'  : year the hurricane occurred (misleading name in schema)
    #   'ind'        : normalized property damage (used to compute alternative outcome)

    # Make sure numeric columns are numeric (coerce errors to NaN)
    numeric_cols = ['name', 'elapsedyrs', 'ndam15', 'wind', 'min', 'masfem', 'alldeaths', 'ind']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the primary variables needed for the main analysis
    required_cols = ['name', 'ndam15', 'wind', 'min', 'masfem', 'alldeaths', 'elapsedyrs']
    existing_required = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=existing_required)

    # Create modeling variables with clear names
    # Continuous femininity index (raw)
    df['name_fem'] = df['name']

    # Binary label from archival coding: 1 if name coded female, 0 otherwise
    df['is_female_name'] = df['elapsedyrs'].astype(int)

    # Dependent variable: log( deaths + 1 ) to reduce skew and handle zeros
    df['log_deaths'] = np.log1p(df['ndam15'].fillna(0))

    # Alternative DV (not used as primary in the default model but kept for sensitivity checks)
    if 'ind' in df.columns:
        df['log_damage'] = np.log1p(df['ind'].fillna(0))

    # Controls: rename for clarity
    df['max_wind'] = df['wind']
    df['min_pressure'] = df['min']
    df['saffir_cat'] = df['masfem']
    # Year of storm
    df['Year'] = df['alldeaths']

    # Standardize continuous predictors (z-score using population std ddof=0 to keep consistency)
    z_cols = ['name_fem', 'max_wind', 'min_pressure', 'saffir_cat', 'Year']
    for c in z_cols:
        if c in df.columns:
            col = df[c]
            # If constant or NA, fill with 0 after transform to avoid NaNs
            mean = col.mean()
            std = col.std(ddof=0)
            if pd.isna(std) or std == 0:
                df[c + '_z'] = 0.0
            else:
                df[c + '_z'] = (col - mean) / std

    # Keep only columns needed for modeling and diagnostics
    keep_cols = [
        'name_fem', 'name_fem_z', 'is_female_name',
        'ndam15', 'log_deaths',
        'max_wind', 'max_wind_z', 'min_pressure', 'min_pressure_z',
        'saffir_cat', 'saffir_cat_z', 'Year', 'Year_z'
    ]
    if 'log_damage' in df.columns:
        keep_cols.append('log_damage')

    # Some of these columns may not exist if dataset is incomplete; subset safely
    keep_cols_existing = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols_existing]

    # Final drop of any rows with NA in the variables actually used in the canonical model
    needed_for_model = ['name_fem_z', 'is_female_name', 'log_deaths', 'max_wind_z', 'min_pressure_z', 'saffir_cat_z', 'Year_z']
    needed_existing = [c for c in needed_for_model if c in df.columns]
    df = df.dropna(subset=needed_existing)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    import statsmodels.formula.api as smf

    # Expected columns in df (from transform):
    # 'log_deaths' (DV), 'name_fem_z' (IV continuous, standardized), 'is_female_name' (IV binary),
    # 'saffir_cat_z' (moderator / control standardized), 'max_wind_z', 'min_pressure_z', 'Year_z'

    # Check that required columns exist
    required = ['log_deaths', 'name_fem_z', 'is_female_name', 'saffir_cat_z', 'max_wind_z', 'min_pressure_z', 'Year_z']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Primary model: test whether more feminine names are associated with higher log deaths
    # (i.e., fewer precautions) controlling for storm intensity and year.
    # Include an interaction between name femininity and Saffir-Simpson category to test moderation.
    formula = 'log_deaths ~ name_fem_z * saffir_cat_z + is_female_name + max_wind_z + min_pressure_z + Year_z'

    model_res = smf.ols(formula=formula, data=df).fit()

    # Return the fitted model object (has .summary(), .params, .bse, etc.)
    return model_res


