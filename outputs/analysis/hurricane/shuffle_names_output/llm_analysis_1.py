from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Attempt to read a CSV at module import time as in original code (harmless if path invalid)
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/shuffle_names_output/hurricane.csv')
except Exception:
    df = pd.DataFrame()


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Coerce and create standardized column names used in modeling
    # 'NameFem' = femininity index (continuous)
    df['NameFem'] = pd.to_numeric(df.get('name'), errors='coerce')

    # elapsedyrs is described as binary 0 (male) / 1 (female)
    # make sure it's numeric 0/1
    if 'elapsedyrs' in df.columns:
        df['FemaleName'] = pd.to_numeric(df['elapsedyrs'], errors='coerce')
    else:
        # if column not present, fill with NaN
        df['FemaleName'] = np.nan

    # Deaths: use ndam15 per schema (total deaths caused by the hurricane)
    df['Deaths'] = pd.to_numeric(df.get('ndam15'), errors='coerce')

    # Create a logged deaths variable for descriptive/diagnostic use
    # Use log1p to handle zeros safely
    df['LogDeaths'] = np.log1p(df['Deaths'])

    # Controls: wind, min (pressure), category, year, and a damage proxy
    df['Wind'] = pd.to_numeric(df.get('wind'), errors='coerce')
    df['MinPressure'] = pd.to_numeric(df.get('min'), errors='coerce')

    # Category: Saffir-Simpson category; ensure a categorical dtype.
    if 'masfem' in df.columns:
        df['Category'] = pd.Categorical(df['masfem'])
    else:
        # create a categorical column of the same length filled with NA
        df['Category'] = pd.Categorical([pd.NA] * len(df))

    # 'alldeaths' is documented as the year column in the provided schema
    df['Year'] = pd.to_numeric(df.get('alldeaths'), errors='coerce')

    # 'ind' is given as normalized damage in the schema; log-transform for modeling
    # fill missing with 0 (no damage) before logging
    df['LogDamage'] = np.log1p(pd.to_numeric(df.get('ind'), errors='coerce').fillna(0))

    # Drop rows that are missing the essential modeling variables
    required = ['NameFem', 'FemaleName', 'Deaths', 'Wind', 'MinPressure', 'Year']
    df = df.dropna(subset=required)

    # Ensure integer type for Deaths (counts) where possible
    # Use astype with errors='ignore' in case values cannot be converted cleanly
    try:
        df['Deaths'] = df['Deaths'].astype(int)
    except Exception:
        # if conversion fails, coerce via to_numeric then astype int
        df['Deaths'] = pd.to_numeric(df['Deaths'], errors='coerce').fillna(0).astype(int)

    # Keep relevant columns only (but do not drop others in case user wants them);
    # this ensures the final dataframe includes all variables referenced in the model
    final_cols = [
        'NameFem', 'FemaleName', 'Deaths', 'LogDeaths',
        'Wind', 'MinPressure', 'Category', 'Year', 'LogDamage',
        # keep some identifiers for diagnostics
        'ndam', 'source', 'ndam15'
    ]

    # Add any missing final columns (if source/ndam not present) with appropriate length
    for c in final_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Ensure final frame has the specified column order
    return df[final_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Work on a copy
    df = df.copy()

    # If there are no observations, return None rather than letting statsmodels error
    if df.shape[0] == 0:
        return None

    # Create design matrix: continuous IV (NameFem), binary IV (FemaleName), controls
    # Convert categorical 'Category' into dummies (drop first to avoid collinearity)
    cat_dummies = pd.get_dummies(df['Category'], prefix='Cat', drop_first=True)

    # Ensure base columns exist in df (they should per transform contract)
    X_base_cols = ['NameFem', 'FemaleName', 'Wind', 'MinPressure', 'Year', 'LogDamage']
    X_base = df[X_base_cols].copy()

    # Convert all predictors to numeric, coercing non-numeric to NaN
    X_base = X_base.apply(pd.to_numeric, errors='coerce')

    # Concatenate with categorical dummies (if any)
    X = pd.concat([X_base, cat_dummies], axis=1)

    # Add constant for intercept
    X = sm.add_constant(X, has_constant='add')

    # Dependent variable (counts)
    y = pd.to_numeric(df['Deaths'], errors='coerce')

    # Drop any rows with missing values in X or y
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    # If no observations remain after dropping missing data, return None
    if X.shape[0] == 0:
        return None

    # Fit a Negative Binomial GLM (appropriate for overdispersed count data).
    model_glm = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    try:
        results = model_glm.fit()
    except Exception:
        # If fit fails for any reason, return None instead of raising
        return None

    # Return results with robust (heteroskedasticity-consistent) standard errors
    try:
        results_robust = results.get_robustcov_results(cov_type='HC3')
    except Exception:
        # if robust cov cannot be computed for some reason, return the original results
        results_robust = results

    return results_robust