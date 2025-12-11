from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataset into a modeling-ready dataframe.

    Final columns produced and used in the model:
    - ndam15: raw fatalities count (numeric)
    - femininity_z: standardized femininity score of the hurricane name (z-score)
    - female_name_bin: binary indicator (0/1) for female name label (elapsedyrs column)
    - wind: numeric storm wind measure
    - min: numeric pressure / storm measure
    - masfem: numeric Saffir-Simpson category
    - ind: raw property damage (kept for transformation)
    - log_ind: log(ind + 1) property damage
    - year: numeric year of the event (taken from 'alldeaths' when present or 'year')

    Missing-value strategy:
    - 'ndam15' is required; if not present we try reasonable alternative column names before raising.
    - 'femininity_z' will always be created; if the underlying femininity measure is missing or constant,
      we fill femininity_z with 0.0 to avoid dropping all observations.
    - Other control columns are created if absent and missing values are filled with sensible defaults
      (column median when available, otherwise 0).
    """
    df = df.copy()

    # Ensure numeric columns are numeric where appropriate
    for col in ['ndam15', 'name', 'wind', 'min', 'masfem', 'ind', 'elapsedyrs']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Construct a sensible 'year' column: prefer 'alldeaths' (dataset notes indicate this is the year),
    # otherwise fall back to an existing 'year' column
    if 'alldeaths' in df.columns:
        df['year'] = pd.to_numeric(df['alldeaths'], errors='coerce')
    elif 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
    else:
        # create the column to satisfy modeling contract; will fill missing values later
        df['year'] = np.nan

    # Dependent variable: ensure fatalities count is numeric and present
    if 'ndam15' not in df.columns:
        # Try to find a reasonable alternative column name for fatalities
        candidates = [c for c in df.columns if any(k in c.lower() for k in ['death', 'fatal', 'ndam'])]
        if candidates:
            df['ndam15'] = pd.to_numeric(df[candidates[0]], errors='coerce')
        else:
            raise KeyError("Expected column 'ndam15' (fatalities) not found in dataframe")
    else:
        df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')

    # Independent variable: femininity score.
    # Prefer 'name' (as documented), but accept an explicit 'femininity' column if present.
    if 'name' in df.columns:
        df['femininity'] = pd.to_numeric(df['name'], errors='coerce')
    elif 'femininity' in df.columns:
        df['femininity'] = pd.to_numeric(df['femininity'], errors='coerce')
    else:
        # No measured femininity available; create a placeholder column of NaN which we will
        # convert to a neutral standardized score (0.0) below to avoid dropping all rows.
        df['femininity'] = np.nan

    # Standardize (z-score). If constant or all NaN, fill with 0.0 to preserve observations.
    fert_std = df['femininity'].std(ddof=0)
    fert_mean = df['femininity'].mean()
    if pd.isna(fert_std) or fert_std == 0:
        # No variation or all missing: impute neutral standardized score 0.0 for all rows
        df['femininity_z'] = 0.0
    else:
        df['femininity_z'] = (df['femininity'] - fert_mean) / fert_std
        # For any remaining NaNs (e.g., entries where femininity was missing), fill with 0.0
        df['femininity_z'] = df['femininity_z'].fillna(0.0)

    # Binary female-name indicator (dataset column 'elapsedyrs' is documented as 0=male name, 1=female name)
    if 'elapsedyrs' in df.columns:
        # coerce to numeric, fillna with 0 (assume non-female if missing), then int
        df['female_name_bin'] = pd.to_numeric(df['elapsedyrs'], errors='coerce').fillna(0).astype(int)
    else:
        # if the dataset doesn't have binary label, create it from femininity threshold (fallback)
        # Use 0 when femininity is NaN (we already filled femininity_z with 0.0 when missing)
        fem_med = df['femininity'].median(skipna=True)
        if pd.isna(fem_med):
            df['female_name_bin'] = 0
        else:
            df['female_name_bin'] = (df['femininity'] > fem_med).astype(int)

    # Log-transform property damage (ind) as an additional severity control. Fill missing damages with 0 before log.
    if 'ind' in df.columns:
        df['ind'] = df['ind'].fillna(0).astype(float)
        df['log_ind'] = np.log(df['ind'] + 1)
    else:
        # create log_ind column filled with zeros so we don't drop observations later
        df['log_ind'] = 0.0

    # Ensure control numeric columns exist; coerce to numeric
    for c in ['wind', 'min', 'masfem']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            # create the column with NaNs to be filled below
            df[c] = np.nan

    # Ensure year is numeric already (created above if missing)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    # Fill missing control values with column medians where possible, otherwise with 0.
    for c in ['wind', 'min', 'masfem', 'log_ind', 'year']:
        if c not in df.columns:
            df[c] = 0.0
            continue
        # compute median excluding NaN
        try:
            med = float(pd.Series(df[c]).median(skipna=True))
        except Exception:
            med = None
        if pd.isna(med):
            # if no valid median (all NaN), fill with 0
            df[c] = df[c].fillna(0.0)
        else:
            df[c] = df[c].fillna(med)

    # Ensure female_name_bin is integer and has no missing values
    if 'female_name_bin' in df.columns:
        df['female_name_bin'] = pd.to_numeric(df['female_name_bin'], errors='coerce').fillna(0).astype(int)
    else:
        df['female_name_bin'] = 0

    # At minimum, require ndam15 to be present; femininity_z has been created and filled to avoid dropping all rows.
    # Drop rows missing ndam15 only.
    df = df.dropna(subset=['ndam15'])

    # Final check / cast types for modeling
    df['ndam15'] = df['ndam15'].astype(float)
    df['femininity_z'] = df['femininity_z'].astype(float)
    df['log_ind'] = df['log_ind'].astype(float)
    df['wind'] = df['wind'].astype(float)
    df['min'] = df['min'].astype(float)
    df['masfem'] = df['masfem'].astype(float)
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(float)
    df['female_name_bin'] = df['female_name_bin'].astype(int)

    # Return dataframe that contains all columns used in the model
    final_cols = ['ndam15', 'femininity_z', 'female_name_bin', 'wind', 'min', 'masfem', 'log_ind', 'year']
    # Ensure these columns exist in the returned dataframe (they should)
    for c in final_cols:
        if c not in df.columns:
            # As a last resort create with zeros
            if c == 'female_name_bin':
                df[c] = 0
            else:
                df[c] = 0.0

    return df[final_cols].copy()


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a Negative Binomial regression of hurricane fatalities on name femininity
    controlling for storm severity and time trends.

    Model specification (primary):
      ndam15 ~ femininity_z + female_name_bin + wind + min + masfem + log_ind + year

    Returns a statsmodels results object with robust covariance estimates (HC3).
    """
    # Ensure required columns are present
    required = ['ndam15', 'femininity_z', 'female_name_bin', 'wind', 'min', 'masfem', 'log_ind', 'year']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(f"Missing required columns for modeling: {missing}")

    # Drop any remaining rows with missing values in required columns to avoid empty arrays
    df_model = df.dropna(subset=required).copy()

    # Ensure there is at least one observation
    if df_model.shape[0] == 0:
        raise ValueError("No observations available for modeling after dropping missing values on required columns.")

    # Prepare dependent and independent variables
    y = df_model['ndam15'].astype(float)
    X = df_model[['femininity_z', 'female_name_bin', 'wind', 'min', 'masfem', 'log_ind', 'year']].astype(float)

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Fit Negative Binomial GLM to account for over-dispersion in counts
    glm_nb = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    res = glm_nb.fit()

    # Compute robust (HC3) covariance for standard errors
    try:
        res_robust = res.get_robustcov_results(cov_type='HC3')
    except Exception:
        # If robust covariance fails for some reason, return the original fit
        res_robust = res

    # Return the fitted (robust) results object
    return res_robust