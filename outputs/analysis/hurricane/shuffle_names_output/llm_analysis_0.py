from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Top-level read (kept for compatibility with original script usage)
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/shuffle_names_output/hurricane.csv')


def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Coerce key columns to numeric where appropriate; create columns if absent.
    # 'name' is the femininity index (continuous)
    if 'name' in df.columns:
        df['name'] = pd.to_numeric(df['name'], errors='coerce')
    else:
        df['name'] = np.nan

    # total deaths (count)
    if 'ndam15' in df.columns:
        df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')
    else:
        df['ndam15'] = np.nan

    # max wind (named 'min' in dataset) -> we'll create final 'max_wind' later
    if 'min' in df.columns:
        df['min'] = pd.to_numeric(df['min'], errors='coerce')
    else:
        df['min'] = np.nan

    # normalized property damage
    if 'ind' in df.columns:
        df['ind'] = pd.to_numeric(df['ind'], errors='coerce')
    else:
        df['ind'] = np.nan

    # elapsedyrs in dataset is a binary gender indicator (0 male, 1 female)
    if 'elapsedyrs' in df.columns:
        df['elapsedyrs'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')
    else:
        df['elapsedyrs'] = np.nan

    # used as event year per schema
    if 'alldeaths' in df.columns:
        df['alldeaths'] = pd.to_numeric(df['alldeaths'], errors='coerce')
    else:
        df['alldeaths'] = np.nan

    # Ensure category is string (will be converted to dummies later)
    if 'category' in df.columns:
        df['category'] = df['category'].fillna('Unknown').astype(str)
    else:
        df['category'] = 'Unknown'

    # Only drop rows missing the primary dependent variable:
    # ndam15 (DV) is required. For 'name' (IV), we'll impute missing values to preserve rows.
    df = df.dropna(subset=['ndam15'])

    # Fill defaults for controls if missing so we don't accidentally drop all rows.
    # For max wind (min), use 0 if missing (conservative).
    df['min'] = df['min'].fillna(0.0)
    # For damage (ind), use 0 if missing (no reported damage).
    df['ind'] = df['ind'].fillna(0.0)
    # event_year from alldeaths: fill with median year if available else 0
    if df['alldeaths'].notna().any():
        median_year = int(df['alldeaths'].median(skipna=True))
    else:
        median_year = 0
    df['alldeaths'] = df['alldeaths'].fillna(median_year)

    # Binary indicator from elapsedyrs column (if absent, default to 0)
    if df['elapsedyrs'].notna().any():
        df['elapsedyrs'] = df['elapsedyrs'].fillna(0)
        # Clip to 0/1 and cast to int
        df['elapsedyrs'] = df['elapsedyrs'].astype(int).clip(0, 1)
    else:
        df['elapsedyrs'] = 0

    # For the femininity index 'name', impute missing values with the mean (so rows are retained),
    # then standardize to create 'name_z'.
    if df['name'].notna().any():
        name_mean = df['name'].mean(skipna=True)
    else:
        # If no valid 'name' values exist, assume mean 0 for imputation (neutral)
        name_mean = 0.0
    df['name'] = df['name'].fillna(name_mean)

    name_mean = df['name'].mean()
    name_std = df['name'].std(ddof=0)
    if pd.isna(name_std) or name_std == 0:
        name_std = 1.0
    df['name_z'] = (df['name'] - name_mean) / name_std

    # Ensure ndam15 is integer where possible (counts)
    df['ndam15'] = pd.to_numeric(df['ndam15'], errors='coerce')
    # Round to nearest integer for counts and ensure non-negative
    df['ndam15'] = df['ndam15'].round().clip(lower=0).astype(int)

    # For downstream use, logged version for diagnostics
    df['log_ndam15'] = np.log1p(df['ndam15'].astype(float))

    # Rename / copy columns to analysis-friendly names (these final names are fixed by contract)
    df['max_wind'] = df['min'].astype(float)
    # Ensure non-negative before log1p
    df['log_damage_ind'] = np.log1p(df['ind'].clip(lower=0).astype(float))
    # event_year from alldeaths per schema
    df['event_year'] = df['alldeaths'].round().astype(int)

    # Binary indicator from elapsedyrs column
    df['is_female_name'] = df['elapsedyrs'].astype(int).clip(0, 1)

    # Ensure category is string
    df['category'] = df['category'].fillna('Unknown').astype(str)

    # Final check: drop any rows that still have NA in variables used in the model
    needed_cols = ['name_z', 'ndam15', 'max_wind', 'log_damage_ind', 'event_year', 'is_female_name', 'category']
    df = df.dropna(subset=needed_cols)

    # Ensure correct dtypes for final dataframe columns
    df['name_z'] = pd.to_numeric(df['name_z'], errors='coerce').astype(float)
    df['ndam15'] = df['ndam15'].astype(int)
    df['max_wind'] = pd.to_numeric(df['max_wind'], errors='coerce').astype(float)
    df['log_damage_ind'] = pd.to_numeric(df['log_damage_ind'], errors='coerce').astype(float)
    df['event_year'] = pd.to_numeric(df['event_year'], errors='coerce').astype(int)
    df['is_female_name'] = df['is_female_name'].astype(int)
    df['category'] = df['category'].astype(str)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


def model(df: pd.DataFrame) -> Any:
    # Work on a copy
    df = df.copy()

    # Verify required final columns exist
    required_cols = {'name_z', 'ndam15', 'max_wind', 'log_damage_ind', 'event_year', 'is_female_name', 'category'}
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Transformed dataframe is missing required columns: {missing}")

    # Check that there is at least one observation to fit the model
    if df.shape[0] == 0:
        raise ValueError("Transformed dataframe contains zero rows; cannot fit model.")

    # Prepare predictors (controls + IV). Include category as dummies (drop first to avoid collinearity).
    predictors = ['name_z', 'max_wind', 'log_damage_ind', 'event_year', 'is_female_name']
    # Ensure predictor columns are present and numeric
    X = df[predictors].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # If any predictor column has become all-NA, raise an informative error
    all_na_cols = [c for c in X.columns if X[c].isna().all()]
    if all_na_cols:
        raise ValueError(f"The following predictor columns contain only NA after coercion: {all_na_cols}")

    # Add categorical dummies for 'category'
    cat_dummies = pd.get_dummies(df['category'].astype(str), prefix='cat', drop_first=True)
    if not cat_dummies.empty:
        # ensure dummies are numeric
        cat_dummies = cat_dummies.astype(float)
        X = pd.concat([X, cat_dummies], axis=1)

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Outcome: count of deaths
    y = pd.to_numeric(df['ndam15'], errors='coerce')

    # Final checks before model fitting
    if X.shape[1] == 0:
        raise ValueError("No predictor columns available to fit the model.")
    if X.isna().any().any() or pd.isna(y).any():
        # Align and drop any rows with NA in X or y
        combined = pd.concat([X, y.rename('ndam15')], axis=1)
        combined = combined.dropna()
        if combined.shape[0] == 0:
            raise ValueError("No complete cases available to fit the model after dropping NA.")
        y = combined['ndam15']
        X = combined.drop(columns=['ndam15'])

    # Ensure numeric numpy arrays
    X_np = np.asarray(X, dtype=float)
    y_np = np.asarray(y, dtype=float)

    # Fit a Negative Binomial GLM to account for over-dispersion in counts.
    model_nb = sm.GLM(y_np, X_np, family=sm.families.NegativeBinomial())
    results_nb = model_nb.fit()

    # Return the fitted results object (has summary(), params, pvalues, etc.)
    return results_nb