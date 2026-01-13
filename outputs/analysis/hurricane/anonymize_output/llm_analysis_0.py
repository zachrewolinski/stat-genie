from typing import Any, Dict, List
import warnings
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _find_and_rename(df: pd.DataFrame, target: str, candidate_keywords: List[List[str]]) -> None:
    """
    If target column not present in df, search for columns whose lowercased names
    match any of the candidate keyword lists (all keywords in a sublist must be present).
    If found, rename the first matching column to target in-place.

    Matching is done on a normalized version of the column name (lowercased,
    with non-alphanumeric characters removed) as well as the original lowercase
    name to handle variants like "min_pressure", "MinPressure", "pressure_mb", etc.
    """
    if target in df.columns:
        return

    cols_lower = {col: col.lower() for col in df.columns}
    cols_normalized = {col: re.sub(r'[^a-z0-9]', '', col.lower()) for col in df.columns}

    for keywords in candidate_keywords:
        # ensure keywords are lowercased and stripped
        kws = [kw.strip().lower() for kw in keywords if isinstance(kw, str) and kw.strip()]
        if not kws:
            continue
        for col in df.columns:
            col_low = cols_lower[col]
            col_norm = cols_normalized[col]
            # match if all keywords appear either in the raw lower name or in the normalized name
            if all((kw in col_low) or (kw in col_norm) for kw in kws):
                df.rename(columns={col: target}, inplace=True)
                return


def _coerce_isfemale(series: pd.Series) -> pd.Series:
    """
    Ensure IsFemale is numeric 0/1 where possible.
    Accepts numeric values, booleans, and common string encodings like 'female', 'F', 'male'.
    Returns a numeric Series (float) with NaN where mapping impossible.
    """
    # If already numeric-ish, coerce directly
    coerced = pd.to_numeric(series, errors='coerce')
    # If many non-NaN after coercion, trust numeric coercion
    if coerced.notna().sum() >= (len(series) / 2):
        return coerced.astype(float)

    # Otherwise try to interpret strings
    mapping = {}
    for val in series.dropna().unique():
        if isinstance(val, str):
            v = val.strip().lower()
            if v in {'female', 'f', 'woman', 'w', '1', 'true', 't', 'yes', 'y'}:
                mapping[val] = 1.0
            elif v in {'male', 'm', 'man', '0', 'false', 'no', 'n'}:
                mapping[val] = 0.0
            else:
                # heuristic: if contains 'fem' -> female; if contains 'mal' or 'male' -> male
                if 'fem' in v:
                    mapping[val] = 1.0
                elif 'mal' in v or 'male' in v:
                    mapping[val] = 0.0
                elif v in {'m', 'f'}:
                    mapping[val] = 1.0 if v == 'f' else 0.0
                else:
                    mapping[val] = np.nan
        elif isinstance(val, (bool, np.bool_)):
            mapping[val] = 1.0 if bool(val) else 0.0
        else:
            # numeric-like but not captured earlier
            try:
                num = float(val)
                if num in (0.0, 1.0):
                    mapping[val] = num
                else:
                    mapping[val] = np.nan
            except Exception:
                mapping[val] = np.nan

    return series.map(mapping).astype(float)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into a cleaned dataframe with columns used in the models.

    Renames feature columns to meaningful names, coerces types, drops rows missing critical fields,
    standardizes continuous predictors, and creates log outcomes for robustness analyses.
    """
    df = df.copy()

    # Primary explicit rename map (common anonymized names -> target final names)
    rename_map = {
        'feature1': 'StormID',
        'feature2': 'Year',
        'feature3': 'Name',
        'feature4': 'MasFem',        # masculinity-femininity index (higher = more feminine)
        'feature5': 'MinPressure',   # minimum pressure at landfall
        'feature6': 'IsFemale',      # binary name gender (0 male, 1 female)
        'feature7': 'Category',      # Saffir-Simpson category
        'feature8': 'Deaths',        # total number of deaths
        'feature9': 'Damage2013',
        'feature10': 'YearsSince',
        'feature11': 'Source',
        'feature12': 'MTurkMasFem',
        'feature13': 'MaxWind',      # max wind speed
        'feature14': 'Damage2015'
    }
    # Apply explicit renames for any matching keys
    existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing_renames:
        df = df.rename(columns=existing_renames)

    # If some required conceptual columns are still missing, attempt to detect likely columns by keywords
    # Define candidate keyword lists for each required target column
    candidate_searches = {
        'MasFem': [['mas', 'fem'], ['masculin', 'femin'], ['mas'], ['fem'], ['masfem'], ['mas_fem'], ['masfem_mturk']],
        'IsFemale': [['is', 'female'], ['isfemale'], ['female'], ['gender'], ['sex'], ['sex_label']],
        'Deaths': [['death'], ['fatalit'], ['fatal'], ['deaths'], ['numdeaths'], ['fatalities']],
        'MaxWind': [['max', 'wind'], ['wind', 'speed'], ['wind'], ['maxwind'], ['windmph'], ['windspeed']],
        'MinPressure': [['min', 'press'], ['min', 'pressure'], ['pressure'], ['press'], ['minpressure'], ['pressuremb'], ['pressure_mb'], ['min']],
        'Year': [['year'], ['yr'], ['season']],
        'Category': [['category'], ['cat'], ['saffir'], ['saffir-simpson'], ['saffir_simpson']],
        'Source': [['source'], ['src'], ['data', 'source']]
    }

    for target, keywords in candidate_searches.items():
        if target not in df.columns:
            _find_and_rename(df, target, keywords)

    # After attempts, check that at least the required columns exist (as columns in final dataframe)
    required = ['MasFem', 'IsFemale', 'Deaths', 'MaxWind', 'MinPressure', 'Year']
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        # Provide a more informative hint by listing available columns
        raise ValueError(
            f"transform: missing required columns after renaming/searching: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Coerce numeric columns and basic cleaning
    numeric_cols = ['Year', 'MasFem', 'MinPressure', 'IsFemale', 'Category', 'Deaths', 'MaxWind', 'Damage2015']
    for col in numeric_cols:
        if col in df.columns and col != 'IsFemale':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Special handling for IsFemale: try to coerce to numeric 0/1 using heuristics
    if 'IsFemale' in df.columns:
        df['IsFemale'] = _coerce_isfemale(df['IsFemale'])

    # Standardize column name casing for optional categorical/source fields if present
    # Ensure Category and Source have exact final names
    # (They may have been detected/renamed above, but handle common lowercased versions)
    if 'category' in df.columns and 'Category' not in df.columns:
        df = df.rename(columns={'category': 'Category'})
    if 'source' in df.columns and 'Source' not in df.columns:
        df = df.rename(columns={'source': 'Source'})

    # Keep only rows with the essential variables present
    df = df.dropna(subset=required)

    # Standardize continuous predictors used as controls and the masculinity-femininity index
    # Use population std (ddof=0) as before, avoid division by zero
    def _zscore(series: pd.Series) -> pd.Series:
        mean = series.mean()
        std = series.std(ddof=0)
        if pd.isna(std) or std == 0:
            return series - mean
        return (series - mean) / std

    df['MasFem_z'] = _zscore(df['MasFem'].astype(float))
    df['MinPressure_z'] = _zscore(df['MinPressure'].astype(float))
    df['MaxWind_z'] = _zscore(df['MaxWind'].astype(float))

    # Center year to improve interpretability
    df['year_cent'] = df['Year'].astype(float) - df['Year'].astype(float).mean()

    # Log-transformed outcomes for robustness checks (log(1 + x))
    df['log_deaths'] = np.log1p(df['Deaths'].fillna(0))
    if 'Damage2015' in df.columns:
        df['log_damage2015'] = np.log1p(pd.to_numeric(df['Damage2015'], errors='coerce').fillna(0))
    else:
        df['log_damage2015'] = np.nan

    # Ensure categorical fields are categorical dtype (if present)
    if 'Category' in df.columns:
        df['Category'] = df['Category'].astype('category')
    if 'Source' in df.columns:
        df['Source'] = df['Source'].astype('category')

    # Final dataframe returned contains the columns described in cvars plus convenient transformations
    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit the primary statistical models to test whether feminine hurricane names are associated
    with differences in fatalities after controlling for storm intensity and year.

    Primary model: Negative Binomial regression of raw death counts on femininity measures and controls.
    Robustness: OLS on log(1 + deaths).

    Returns a dictionary with fitted model results objects (or robustified equivalents).
    """
    # Work on a copy to avoid modifying input
    df = df.copy()

    # Predictor columns to include
    base_cols = ['MasFem_z', 'IsFemale', 'MinPressure_z', 'MaxWind_z', 'year_cent']

    # Verify required model input columns are present
    missing = [col for col in base_cols + ['Deaths', 'log_deaths'] if col not in df.columns]
    if missing:
        raise ValueError(f"model: missing required columns in input dataframe: {missing}")

    # Build category dummies for the Saffir-Simpson Category (if present)
    if 'Category' in df.columns:
        cat_dummies = pd.get_dummies(df['Category'], prefix='cat', drop_first=True)
    else:
        cat_dummies = pd.DataFrame(index=df.index)

    # Build source dummies to control for reporting differences across sources
    if 'Source' in df.columns:
        source_dummies = pd.get_dummies(df['Source'], prefix='src', drop_first=True)
    else:
        source_dummies = pd.DataFrame(index=df.index)

    # Concatenate design matrix (only the conceptual variables and their dummies)
    X = pd.concat([df[base_cols], cat_dummies, source_dummies], axis=1)

    # Drop any rows with missing values in the design matrix or outcome
    model_df = pd.concat([df[['Deaths', 'log_deaths']], X], axis=1).dropna()
    if model_df.shape[0] == 0:
        raise ValueError("model: no rows available after dropping missing values; cannot fit models.")

    X_clean = model_df[X.columns]
    y_deaths = model_df['Deaths']
    y_log_deaths = model_df['log_deaths']

    # Add constant
    X_design = sm.add_constant(X_clean, has_constant='add')

    results: Dict[str, Any] = {}

    # 1) Negative Binomial regression for count outcome (Deaths)
    try:
        nb_model = sm.GLM(y_deaths, X_design, family=sm.families.NegativeBinomial()).fit()
        results['nb_model'] = nb_model
    except Exception as e:
        # If NegativeBinomial fails, fall back to Poisson with robust SEs
        warnings.warn(f"NegativeBinomial failed ({e}); falling back to Poisson with robust SEs.")
        poisson_res = sm.GLM(y_deaths, X_design, family=sm.families.Poisson()).fit()
        poisson_robust = poisson_res.get_robustcov_results(cov_type='HC0')
        results['nb_model'] = poisson_robust
        results['nb_model_warning'] = f'NegativeBinomial failed, used Poisson instead: {str(e)}'

    # 2) OLS on log(1 + deaths) as robustness
    ols_res = sm.OLS(y_log_deaths, X_design).fit()
    ols_robust = ols_res.get_robustcov_results(cov_type='HC3')
    results['ols_log_deaths'] = ols_robust

    # Return the fitted statsmodels results objects for downstream inspection
    return results


# If this file is run directly, provide a small sanity check (won't run during import)
if __name__ == "__main__":
    # Minimal test to ensure functions run (with synthetic data)
    test_df = pd.DataFrame({
        'feature4': [0.2, 0.8, 0.5],
        'feature6': [0, 1, 1],
        'feature8': [10, 0, 3],
        'feature13': [80, 120, 60],
        'feature5': [950, 940, 960],
        'feature2': [2000, 2005, 2010],
        'feature7': ['1', '2', '1'],
        'feature11': ['A', 'B', 'A']
    })
    tr = transform(test_df)
    res = model(tr)
    print("Sanity check completed. Models keys:", list(res.keys()))