from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

# If present in the environment, this read can be left; otherwise it's harmless.
# The functions below operate on DataFrame inputs and do not rely on this variable.
try:
    df = pd.read_csv('/accounts/campus/austin.zane/stat-genie/.venv/lib/python3.11/site-packages/blade_bench/datasets/hurricane/data.csv')
except Exception:
    df = pd.DataFrame()


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure expected raw columns exist in the DataFrame (create if missing)
    # Note: 'name' is kept as-is (may be numeric rating or string); we'll handle numeric conversion for name separately.
    expected_cols = ['ndam15', 'name', 'elapsedyrs', 'wind', 'min', 'masfem', 'alldeaths']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Convert relevant columns to numeric where appropriate (coerce errors to NaN)
    # Exclude 'name' here to avoid accidentally turning string hurricane names into NaN;
    # we'll explicitly convert 'name' to numeric for name_c creation below.
    numeric_cols = ['ndam15', 'elapsedyrs', 'wind', 'min', 'masfem', 'alldeaths']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing the dependent variable:
    # - ndam15 is required for log_deaths (DV)
    df = df.dropna(subset=['ndam15'])

    # If after dropping there are no rows, return an empty DataFrame with the required final columns
    final_cols = ['ndam15', 'log_deaths', 'name', 'name_c', 'elapsedyrs', 'wind', 'min', 'masfem', 'alldeaths']
    if df.shape[0] == 0:
        # Create an empty DataFrame with the required columns and appropriate dtypes
        empty_df = pd.DataFrame({c: pd.Series(dtype=float) for c in final_cols})
        # elapsedyrs should be integer dtype when possible
        empty_df['elapsedyrs'] = empty_df['elapsedyrs'].astype('Int64')
        return empty_df

    # Create dependent variable: log-transformed deaths to reduce skew
    # Add 1 to allow zero-death storms
    df['log_deaths'] = np.log(df['ndam15'] + 1)

    # Create numeric version of 'name' for constructing name_c (the continuous femininity rating).
    # If 'name' is already numeric, this will keep values; if not, will be NaN.
    name_num = pd.to_numeric(df['name'], errors='coerce')

    # If there are no numeric name ratings at all, create a default (zeros).
    # Otherwise, impute missing numeric name ratings with the median rating.
    if name_num.notna().sum() == 0:
        name_num = pd.Series(0.0, index=df.index)
    else:
        median_name = name_num.median(skipna=True)
        name_num = name_num.fillna(median_name)

    # Center the continuous name femininity score for interpretability
    df['name_c'] = name_num - name_num.mean()

    # Ensure binary elapsedyrs is 0/1; coerce non-missing values to integers.
    # Fill missing elapsedyrs with 0 (assumes missing means not female-coded).
    df['elapsedyrs'] = df['elapsedyrs'].fillna(0)
    # Some values may be non-binary; coerce to 0/1 by rounding after converting to numeric
    df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce').fillna(0).round().astype(int)

    # For control variables, impute missing values with the column median (numeric) where appropriate.
    # This prevents excessive row loss while preserving the variables in the final DataFrame.
    for c in ['wind', 'min', 'masfem', 'alldeaths']:
        if c in df.columns:
            # If entire column is NaN, fill with 0 to ensure presence
            if df[c].notna().sum() == 0:
                df[c] = 0.0
            else:
                median_val = df[c].median(skipna=True)
                df[c] = df[c].fillna(median_val)

    # Ensure 'name' column exists in final (keep original values; if entirely missing fill with empty string)
    if 'name' not in df.columns:
        df['name'] = ''
    else:
        # If name column is entirely NaN, fill with empty string to avoid NaN in final dataframe
        if df['name'].isna().all():
            df['name'] = ''

    # Final dataframe will include: ndam15, log_deaths, name, name_c, elapsedyrs, wind, min, masfem, alldeaths
    # Ensure all required final columns exist (create if necessary)
    for c in final_cols:
        if c not in df.columns:
            # For safety create missing columns filled with zeros (or appropriate type)
            if c == 'elapsedyrs':
                df[c] = 0
            elif c == 'name':
                df[c] = ''
            else:
                df[c] = 0.0

    # Select and return only the final columns in the required order
    df_final = df[final_cols].copy()

    # Make sure there are no remaining NaNs in numeric final columns by filling with sensible defaults
    numeric_final_cols = ['ndam15', 'log_deaths', 'name_c', 'elapsedyrs', 'wind', 'min', 'masfem', 'alldeaths']
    for c in numeric_final_cols:
        if df_final[c].isna().any():
            if c == 'elapsedyrs':
                df_final[c] = pd.to_numeric(df_final[c], errors='coerce').fillna(0).astype(int)
            else:
                df_final[c] = pd.to_numeric(df_final[c], errors='coerce').fillna(0.0)

    # Ensure elapsedyrs dtype is integer
    df_final['elapsedyrs'] = df_final['elapsedyrs'].astype(int)

    return df_final


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Fit OLS models predicting log_deaths from name femininity and controls.
    # Returns a dict with two fitted models for comparison:
    #  - model_name: continuous name femininity (name_c) as primary IV
    #  - model_binary: binary female-name indicator (elapsedyrs) as primary IV

    results = {}

    # Verify required columns exist
    required_final_cols = ['ndam15', 'log_deaths', 'name_c', 'elapsedyrs', 'wind', 'min', 'masfem', 'alldeaths']
    missing = [c for c in required_final_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns for modeling: {missing}")

    # Ensure numeric types for modeling columns
    model_numeric_cols = ['log_deaths', 'name_c', 'elapsedyrs', 'wind', 'min', 'masfem', 'alldeaths']
    for c in model_numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop any rows with missing outcome or with missing exogenous variables for each model separately
    y = df['log_deaths']

    # Model 1: continuous name score as primary IV
    X_cols_name = ['name_c', 'elapsedyrs', 'wind', 'min', 'masfem', 'alldeaths']
    X1 = df[X_cols_name].copy()
    # Drop rows with missing values in y or X1
    model1_df = pd.concat([y, X1], axis=1).dropna()
    if model1_df.shape[0] == 0:
        raise ValueError("No observations available after dropping missing values for model_name.")
    y1 = model1_df['log_deaths'].astype(float)
    X1_clean = model1_df[X_cols_name].astype(float)
    X1_clean = sm.add_constant(X1_clean, has_constant='add')
    model_name = sm.OLS(y1, X1_clean).fit()
    results['model_name'] = model_name

    # Model 2: binary elapsedyrs as primary IV
    X_cols_bin = ['elapsedyrs', 'wind', 'min', 'masfem', 'alldeaths']
    X2 = df[X_cols_bin].copy()
    model2_df = pd.concat([y, X2], axis=1).dropna()
    if model2_df.shape[0] == 0:
        raise ValueError("No observations available after dropping missing values for model_binary.")
    y2 = model2_df['log_deaths'].astype(float)
    X2_clean = model2_df[X_cols_bin].astype(float)
    X2_clean = sm.add_constant(X2_clean, has_constant='add')
    model_binary = sm.OLS(y2, X2_clean).fit()
    results['model_binary'] = model_binary

    return results