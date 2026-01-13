from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load raw data (path retained from original file; can be overwritten by caller)
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/shuffle_names_output/hurricane.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw hurricane dataframe into the analytic dataframe.

    Produces a dataframe with the exact required final columns:
      - 'deaths', 'femininity_z', 'female_name', 'wind', 'min_pressure',
        'saffir', 'year', 'log_property_damage'

    This function is robust to missing control columns by imputing reasonable
    defaults (medians or zeros) so that downstream modeling does not fail due
    to complete-case filtering on many controls. It requires that the input
    dataframe contain usable values for deaths and the femininity rating; if
    those are missing for all rows the returned dataframe will be empty.
    """
    df = df.copy()

    # Map possible source columns to the canonical final columns.
    # Dependent variable: deaths (try common alternatives)
    if 'ndam15' in df.columns:
        df['deaths'] = pd.to_numeric(df['ndam15'], errors='coerce')
    elif 'ndam' in df.columns:
        df['deaths'] = pd.to_numeric(df['ndam'], errors='coerce')
    else:
        # fallback: try any existing 'deaths' column
        df['deaths'] = pd.to_numeric(df.get('deaths'), errors='coerce')

    # Feminine/masculinity continuous rating: expect 'name' per schema, but allow 'femininity'
    femininity_candidates = ['name', 'femininity', 'femininity_score', 'masfem']
    femininity_col = None
    for col in femininity_candidates:
        if col in df.columns:
            # coerce to numeric; if strings are present this will yield NaN
            series = pd.to_numeric(df[col], errors='coerce')
            # accept if at least one non-NaN value
            if series.notna().any():
                df['femininity'] = series
                femininity_col = col
                break
    if femininity_col is None:
        # as a last resort, attempt to coerce an existing 'name' even if it's string;
        # this will likely be NaN, but keep consistent column name
        df['femininity'] = pd.to_numeric(df.get('name'), errors='coerce')

    # Binary female name indicator (0 = male, 1 = female) expected 'elapsedyrs' per schema,
    # but allow a few alternatives and coerce to 0/1
    female_candidates = ['elapsedyrs', 'female', 'is_female', 'female_name']
    female_col = None
    for col in female_candidates:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors='coerce')
            if series.notna().any():
                df['female_name'] = series
                female_col = col
                break
    if female_col is None:
        # create column if missing (will be filled/imputed later)
        df['female_name'] = pd.NA

    # Controls: wind, min_pressure (from 'min'), saffir (from 'masfem'), year (from 'alldeaths' or 'year'),
    # property_damage (from 'ind')
    df['wind'] = pd.to_numeric(df.get('wind'), errors='coerce')
    df['min_pressure'] = pd.to_numeric(df.get('min'), errors='coerce')
    df['saffir'] = pd.to_numeric(df.get('masfem'), errors='coerce')

    if 'alldeaths' in df.columns:
        df['year'] = pd.to_numeric(df['alldeaths'], errors='coerce')
    else:
        df['year'] = pd.to_numeric(df.get('year'), errors='coerce')

    df['property_damage'] = pd.to_numeric(df.get('ind'), errors='coerce')

    # Require at minimum that deaths and femininity be present (non-missing).
    # These are the essential variables for the analysis. If these are missing,
    # resulting dataframe will be empty.
    df = df[df['deaths'].notna() & df['femininity'].notna()].copy()

    # Ensure non-negative death counts and numeric type
    df['deaths'] = pd.to_numeric(df['deaths'], errors='coerce')
    df = df[df['deaths'].ge(0)]

    # Impute remaining controls so that rows are not dropped later due to missingness.
    # Use medians for continuous controls where available; fall back to sensible defaults.
    # wind
    if df['wind'].notna().any():
        wind_median = float(df['wind'].median(skipna=True))
    else:
        wind_median = 0.0
    df['wind'] = df['wind'].fillna(wind_median).astype(float)

    # min_pressure
    if df['min_pressure'].notna().any():
        min_median = float(df['min_pressure'].median(skipna=True))
    else:
        min_median = 0.0
    df['min_pressure'] = df['min_pressure'].fillna(min_median).astype(float)

    # saffir (ordinal)
    if df['saffir'].notna().any():
        saffir_median = float(df['saffir'].median(skipna=True))
    else:
        saffir_median = 1.0
    # round saffir to nearest integer within 1-5
    df['saffir'] = df['saffir'].fillna(saffir_median).round().clip(lower=1, upper=5).astype(int)

    # year
    if df['year'].notna().any():
        year_median = int(df['year'].median(skipna=True))
    else:
        year_median = 0
    df['year'] = df['year'].fillna(year_median).astype(int)

    # property damage -> log transform
    df['property_damage'] = df['property_damage'].fillna(0.0).astype(float)
    df['log_property_damage'] = np.log1p(df['property_damage'])

    # female_name: coerce to binary 0/1. If values are not present, default to 0.
    df['female_name'] = pd.to_numeric(df['female_name'], errors='coerce').fillna(0)
    # Map anything equal to 1 to 1, everything else to 0
    df['female_name'] = df['female_name'].apply(lambda x: 1 if float(x) == 1.0 else 0).astype(int)

    # Standardize femininity for interpretability (z-score)
    femininity_mean = float(df['femininity'].mean())
    femininity_std = float(df['femininity'].std(ddof=0))
    if (femininity_std == 0) or np.isnan(femininity_std):
        df['femininity_z'] = 0.0
    else:
        df['femininity_z'] = (df['femininity'] - femininity_mean) / femininity_std

    # Final dataframe: keep only the required final columns in the specified names
    final_cols = [
        'deaths', 'femininity_z', 'female_name', 'wind', 'min_pressure',
        'saffir', 'year', 'log_property_damage'
    ]
    # Ensure all final columns exist (they should, given the steps above)
    for col in final_cols:
        if col not in df.columns:
            # create placeholder if somehow missing
            df[col] = 0.0

    df = df[final_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a Negative Binomial regression predicting hurricane fatalities from name femininity
    controlling for storm intensity and year-specific factors.

    Model specification (main):
      deaths ~ femininity_z + female_name + wind + min_pressure + saffir + year + log_property_damage

    Returns the fitted model results object (statsmodels). Also prints summary.
    """
    # Copy to avoid modifying the caller's dataframe
    data = df.copy()

    # Ensure required columns exist
    predictors = [
        'femininity_z', 'female_name', 'wind', 'min_pressure', 'saffir', 'year', 'log_property_damage'
    ]
    missing = [c for c in predictors + ['deaths'] if c not in data.columns]
    if missing:
        raise ValueError(f"Transformed dataframe is missing required columns: {missing}")

    # Drop any rows with missing outcome
    data = data[data['deaths'].notna()].copy()

    # Prepare design matrix
    X = data[predictors].copy()

    # Ensure numeric and no NA in X: impute with column medians if necessary
    for col in X.columns:
        if X[col].isna().any():
            if X[col].dtype.kind in 'biufc':
                median = X[col].median(skipna=True)
                if np.isnan(median):
                    median = 0.0
                X[col] = X[col].fillna(median)
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
        # cast to float for statsmodels
        X[col] = X[col].astype(float)

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    y = pd.to_numeric(data['deaths'], errors='coerce').astype(float)

    # After imputation, ensure there is at least one observation
    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("No observations available after preprocessing for model fitting.")

    # Fit Negative Binomial GLM
    model_glm = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    results = model_glm.fit(maxiter=100, disp=False)

    # Print summary for immediate inspection
    print(results.summary())

    return results