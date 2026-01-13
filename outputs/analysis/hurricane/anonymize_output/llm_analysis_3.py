from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/hurricane/anonymize_output/hurricane.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset (feature1..feature14) to a cleaned dataframe with clearly named columns
    used by the statistical model. The function will:
      - Make a shallow copy of the input
      - Rename relevant columns to meaningful names
      - Drop rows missing the key IV (masfem) or the primary DV (deaths) when present
      - Create log-transformed outcome variables (log_deaths, log_damage)
      - Create a standardized masfem score (masfem_z)
      - Create a female binary indicator from feature6
      - Ensure categorical/source variables are well-formed

    The final dataframe will include the exact column names required by the model contract.
    """
    df = df.copy()

    # Primary intended rename mapping (if featureX columns exist)
    rename_map = {
        'feature1': 'storm_id',
        'feature2': 'year',
        'feature3': 'name',
        'feature4': 'masfem',            # continuous masculinity-femininity index (higher = more feminine)
        'feature5': 'min_pressure',      # minimum pressure at landfall (NOAA)
        'feature6': 'female',            # binary indicator 0=male,1=female
        'feature7': 'category',          # Saffir-Simpson category
        'feature8': 'deaths',            # total number of deaths caused by the hurricane
        'feature9': 'damage_raw',        # damage (normalized to 2013) - alternative
        'feature10': 'years_since',      # number of years elapsed since the hurricane
        'feature11': 'source',           # source of the hurricane data
        'feature12': 'mturk_masfem',     # MTurk rating of masculinity/femininity
        'feature13': 'wind_speed',       # maximum wind speed at landfall
        'feature14': 'damage2015'        # damage normalized to 2015 values (preferred damage measure)
    }

    # Apply the straightforward rename where possible
    cols_to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    if cols_to_rename:
        df = df.rename(columns=cols_to_rename)

    # For robustness, if key target columns are still missing, try to find plausible alternatives
    # (e.g., column names that include 'death', 'fatal', 'damage2015', 'min_pressure', etc.)
    # This is conservative: we only rename the first plausible match for each missing target.
    def find_and_rename(target: str, keywords: List[str]):
        if target in df.columns:
            return
        for col in df.columns:
            low = col.lower()
            if any(kw in low for kw in keywords):
                df.rename(columns={col: target}, inplace=True)
                break

    find_and_rename('deaths', ['death', 'fatal'])
    find_and_rename('damage2015', ['damage2015', 'damage_2015', '2015damage'])
    find_and_rename('damage_raw', ['damage', 'damage_raw', 'property_damage'])
    find_and_rename('masfem', ['masfem', 'mas_fem', 'mascul', 'feminin'])
    find_and_rename('min_pressure', ['pressure', 'min_pressure', 'minimum_pressure'])
    find_and_rename('wind_speed', ['wind', 'wind_speed', 'max_wind'])
    find_and_rename('female', ['female', 'sex', 'gender'])
    find_and_rename('category', ['category', 'saffir', 'saffir-simpson'])
    find_and_rename('year', ['year', 'yr'])
    find_and_rename('source', ['source', 'dataset'])

    # Convert types where needed
    # Ensure numeric columns are numeric
    numeric_cols = ['masfem', 'min_pressure', 'female', 'category', 'deaths', 'damage2015', 'wind_speed', 'year', 'years_since']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing the key independent variable or primary dependent variable (if present)
    required_for_drop = [c for c in ['masfem', 'deaths'] if c in df.columns]
    if required_for_drop:
        df = df.dropna(subset=required_for_drop)

    # Create log-transformed dependent variables to reduce skew and handle zeros
    if 'deaths' in df.columns:
        # We only compute log_deaths for rows where deaths is numeric; keep NaN where deaths is NaN
        df['log_deaths'] = np.where(df['deaths'].notna(), np.log1p(df['deaths']), np.nan)
    else:
        # Create placeholder column (all NaN) so downstream code always finds the column name
        df['log_deaths'] = np.nan

    # Prefer damage normalized to 2015 (feature14 -> damage2015); fall back to damage_raw if needed
    if 'damage2015' in df.columns and df['damage2015'].notna().sum() > 0:
        df['damage_used'] = df['damage2015']
    elif 'damage_raw' in df.columns and df['damage_raw'].notna().sum() > 0:
        df['damage_used'] = df['damage_raw']
    else:
        df['damage_used'] = np.nan

    if 'damage_used' in df.columns:
        df['log_damage'] = np.where(df['damage_used'].notna(), np.log1p(df['damage_used']), np.nan)
    else:
        # ensure a column exists (all NaN)
        df['log_damage'] = np.nan

    # Standardize the masfem score (z-score) to make coefficients comparable across models
    if 'masfem' in df.columns:
        std = df['masfem'].std(ddof=0)
        std = std if std != 0 and not np.isnan(std) else 1.0
        df['masfem_z'] = (df['masfem'] - df['masfem'].mean()) / std
    else:
        df['masfem_z'] = np.nan

    # Ensure female is binary (0/1). If original coding is 0/1 keep; otherwise coerce.
    if 'female' in df.columns:
        def coerce_female(x):
            try:
                if pd.isna(x):
                    return np.nan
                if isinstance(x, (int, np.integer)) and (x == 0 or x == 1):
                    return int(x)
                sx = str(x).strip()
                if sx in {'1', '1.0', 'True', 'true', 'T'}:
                    return 1
                if sx in {'0', '0.0', 'False', 'false', 'F'}:
                    return 0
                # fallback: if numeric-like but not 0/1, threshold at 0.5
                stripped = sx.replace('.', '', 1).lstrip('-')
                if stripped.isdigit():
                    xf = float(sx)
                    return 1 if xf > 0.5 else 0
            except Exception:
                pass
            return np.nan
        df['female'] = df['female'].apply(coerce_female)
    else:
        df['female'] = np.nan

    # If female has missing values, fill from masfem threshold (only if necessary): masfem > median -> female=1
    if df['female'].isna().sum() > 0 and 'masfem' in df.columns and df['masfem'].notna().any():
        median_m = df['masfem'].median()
        df['female'] = df['female'].fillna(df['masfem'].apply(lambda m: 1 if m > median_m else 0))

    # Clean categorical variables
    if 'source' in df.columns:
        df['source'] = df['source'].fillna('unknown').astype(str)

    # category should be treated as numeric/ordinal; ensure integer values where possible
    if 'category' in df.columns:
        df['category'] = pd.to_numeric(df['category'], errors='coerce')

    # Year: numeric, if missing try to impute from years_since and current year (not necessary here) — drop if missing
    if 'year' in df.columns:
        # Only coerce but avoid forcing a drop if 'year' exists but is all NaN
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        if df['year'].notna().any():
            df = df.dropna(subset=['year'])
            # coerce to int safely
            df['year'] = df['year'].astype(int)

    # Keep only columns needed for modeling (but keep a few extras for diagnostics)
    keep_cols = [
        'storm_id', 'year', 'name', 'masfem', 'masfem_z', 'female', 'min_pressure',
        'wind_speed', 'category', 'deaths', 'log_deaths', 'damage2015', 'damage_used', 'log_damage', 'source', 'years_since'
    ]

    # Ensure all required final columns from the contract exist (create placeholders if missing)
    final_required = ['masfem', 'log_deaths', 'female', 'min_pressure', 'wind_speed', 'category', 'year', 'source', 'log_damage']
    for col in final_required:
        if col not in df.columns:
            df[col] = np.nan

    # Now select the keep columns that we want to return (they will now exist, possibly filled with NaN)
    df = df[[c for c in keep_cols if c in df.columns]]

    # Final: drop any remaining rows with NA in any of the core modeling columns, but only require columns that have at least
    # some non-missing values to avoid dropping entire dataset when a column is universally missing.
    core_required = ['masfem_z', 'log_deaths', 'min_pressure', 'wind_speed', 'category', 'year']
    subset = [c for c in core_required if c in df.columns and df[c].notna().any()]
    if subset:
        df = df.dropna(subset=subset)

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> dict:
    """
    Run OLS regressions to estimate the relationship between name femininity and storm impacts.
    Primary model: log_deaths ~ masfem_z + female + intensity controls + year + source fixed effects
    Secondary model: log_damage ~ masfem_z + same controls

    Returns a dictionary with fitted model results (statsmodels RegressionResults objects) or None when a model
    could not be fit due to insufficient data.
    """
    import statsmodels.formula.api as smf

    # Make a local copy to avoid modifying the caller's dataframe
    data = df.copy()

    # Ensure categorical source is treated as categorical
    if 'source' in data.columns:
        data['source'] = data['source'].astype('category')

    # Helper to attempt fitting a formula safely. Returns fitted model or None.
    def safe_fit(formula: str, dataset: pd.DataFrame):
        # Determine lhs and rhs variables to drop NA appropriately
        try:
            lhs, rhs = formula.split('~')
        except Exception:
            lhs = None
            rhs = None
        lhs = lhs.strip() if lhs is not None else None

        # Parse RHS tokens conservatively
        rhs_vars = []
        if rhs is not None:
            tokens = [tok.strip() for tok in rhs.split('+')]
            for tok in tokens:
                if tok == '' or tok in {'1', '0'}:
                    continue
                # Handle categorical notation C(var)
                if tok.startswith('C(') and tok.endswith(')'):
                    inner = tok[2:].strip()
                    inner = inner.strip('() ')
                    rhs_vars.append(inner)
                else:
                    # remove any spaces and function wrappers (simple)
                    cleaned = tok.replace(' ', '')
                    rhs_vars.append(cleaned)

        subset_cols = []
        if lhs:
            subset_cols.append(lhs)
        subset_cols.extend([v for v in rhs_vars if v])

        # Only keep columns that actually exist in dataset
        subset_cols = [c for c in subset_cols if c in dataset.columns]

        # If no columns to check or no rows, skip fitting
        if len(subset_cols) == 0 or dataset.shape[0] == 0:
            return None

        # Drop rows with NA in any of the required columns for this formula
        ds = dataset.dropna(subset=subset_cols)
        if ds.shape[0] < 2:
            # Not enough observations to fit a model
            return None

        try:
            mod = smf.ols(formula, data=ds)
            res = mod.fit(cov_type='HC3')
            return res
        except Exception:
            return None

    # Build formula strings using the exact column names required by the contract.
    # Primary regression: deaths
    formula_deaths = 'log_deaths ~ masfem_z + female + wind_speed + min_pressure + category + year'
    if 'source' in data.columns:
        formula_deaths += ' + C(source)'

    deaths_model = safe_fit(formula_deaths, data)

    # Secondary regression: property damage
    formula_damage = 'log_damage ~ masfem_z + female + wind_speed + min_pressure + category + year'
    if 'source' in data.columns:
        formula_damage += ' + C(source)'

    damage_model = safe_fit(formula_damage, data)

    # For transparency, also return a simple unadjusted model (bivariate) of masfem on log_deaths
    bivar_model = safe_fit('log_deaths ~ masfem_z', data)

    results = {
        'deaths_model': deaths_model,
        'damage_model': damage_model,
        'bivariate_deaths_model': bivar_model,
        'n_obs': int(data.shape[0])
    }

    return results