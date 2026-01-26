from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the original hurricane dataframe into the analysis-ready dataframe.

    Expected input columns: feature1..feature14 (per dataset schema). This function:
      - renames columns to clear names,
      - coerces types where appropriate,
      - creates standardized MasFem (MasFem_z), log-transformed outcomes,
      - creates decade and categorical fields,
      - drops rows missing key variables needed for the primary analyses.

    Returns dataframe containing at least the columns referenced in the conceptual model.
    """
    df = df.copy()

    # Rename original columns to meaningful names
    rename_map = {
        'feature1': 'StormID',
        'feature2': 'Year',
        'feature3': 'Name',
        'feature4': 'MasFem',          # masculinity-femininity index (higher => more feminine)
        'feature5': 'MinPressure',     # minimum pressure at landfall
        'feature6': 'IsFemaleName',    # binary 0/1 indicator (0 male name, 1 female name)
        'feature7': 'Category',        # Saffir-Simpson category
        'feature8': 'Deaths',          # total number of deaths
        'feature9': 'Damage2013',      # damage adjusted to 2013 monetary values
        'feature10': 'YearsSince',     # number of years elapsed since the hurricane
        'feature11': 'Source',         # source of data
        'feature12': 'MTurk_MasFem',   # MTurk rating of masculinity-femininity
        'feature13': 'MaxWind',        # maximum wind speed at landfall
        'feature14': 'Damage2015'      # damage adjusted to 2015 monetary values (kept for completeness)
    }
    df = df.rename(columns=rename_map)

    # Ensure numeric types for expected numeric columns
    numeric_cols = [
        'Year', 'MasFem', 'MinPressure', 'IsFemaleName', 'Category', 'Deaths',
        'Damage2013', 'YearsSince', 'MTurk_MasFem', 'MaxWind', 'Damage2015'
    ]
    for c in numeric_cols:
        if c in df.columns:
            # coerce to numeric; errors -> NaN; keep numpy dtypes (float) rather than pandas nullable ints
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Standardize the MasFem measure (z-score). Keep original MasFem too.
    if 'MasFem' in df.columns:
        masmean = df['MasFem'].mean()
        masstd = df['MasFem'].std(ddof=0)
        if pd.isna(masmean):
            df['MasFem_z'] = np.nan
        else:
            df['MasFem_z'] = (df['MasFem'] - masmean) / (masstd if (masstd != 0 and not pd.isna(masstd)) else 1.0)
    else:
        df['MasFem_z'] = np.nan

    # Coerce IsFemaleName to a standard numeric dtype (0/1) if possible. Keep as float if missing values exist.
    if 'IsFemaleName' in df.columns:
        # Already coerced to numeric above; keep float dtype so patsy/statsmodels can interpret it
        # If there are no missing values and values are integer-like, cast to int64
        if df['IsFemaleName'].notna().all():
            # Round to nearest integer then cast to int
            df['IsFemaleName'] = df['IsFemaleName'].round().astype('int64')
        else:
            df['IsFemaleName'] = df['IsFemaleName'].astype('float64')

    # Create log-transformed dependent variable and an auxiliary outcome for damage
    # Add 1 to avoid log(0) issues
    if 'Deaths' in df.columns:
        df['LogDeaths'] = np.log(df['Deaths'].fillna(0) + 1)
    else:
        df['LogDeaths'] = np.nan

    if 'Damage2013' in df.columns:
        df['LogDamage2013'] = np.log(df['Damage2013'].fillna(0) + 1)
    else:
        df['LogDamage2013'] = np.nan

    # Derive Decade to capture coarse temporal trends
    if 'Year' in df.columns:
        # Compute decade for each row; leave NaN where Year is missing
        def compute_decade(y):
            if pd.isna(y):
                return np.nan
            try:
                yi = int(y)
                return (yi // 10) * 10
            except Exception:
                return np.nan

        df['Decade'] = df['Year'].apply(compute_decade)
        # Use categorical dtype for modeling with C(Decade)
        df['Decade'] = df['Decade'].astype('category')
    else:
        df['Decade'] = pd.Series(index=df.index, dtype='category')

    # Category and Source as categorical for modeling (avoid pandas nullable integer dtype)
    if 'Category' in df.columns:
        # Category may be numeric but treat as categorical fixed effect
        df['Category'] = df['Category'].astype('category')

    if 'Source' in df.columns:
        df['Source'] = df['Source'].astype('category')

    # Ensure YearsSince, MaxWind, MinPressure are numpy numeric types (float64) for modeling
    for col in ('YearsSince', 'MaxWind', 'MinPressure', 'MTurk_MasFem'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

    # Drop rows that are missing the primary independent variable or dependent variable
    required_for_primary = ['MasFem_z', 'LogDeaths', 'MaxWind', 'MinPressure']
    present_required = [c for c in required_for_primary if c in df.columns]
    if present_required:
        df = df.dropna(subset=present_required)

    # Final requested columns (keep extra columns for robustness checks)
    requested_columns = [
        'StormID', 'Year', 'Decade', 'Name', 'MasFem', 'MasFem_z', 'MTurk_MasFem', 'IsFemaleName',
        'Category', 'MaxWind', 'MinPressure', 'YearsSince', 'Source',
        'Deaths', 'LogDeaths', 'Damage2013', 'LogDamage2013', 'Damage2015'
    ]
    # Keep only columns that exist in df
    requested_columns = [c for c in requested_columns if c in df.columns]

    return df[requested_columns]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit OLS models testing whether feminine hurricane names are associated with higher fatalities
    (our proxy for fewer precautionary actions). Returns fitted results objects for the primary
    specification and an auxiliary specification on damage.

    Uses robust (HC3) standard errors to reduce sensitivity to heteroskedasticity.

    Expected columns in df: MasFem_z, IsFemaleName, LogDeaths, LogDamage2013, MaxWind, MinPressure,
    Category, YearsSince, Source, MTurk_MasFem, Decade.
    """
    results = {}

    # Primary model: log fatalities ~ standardized femininity + name gender + storm intensity controls + temporal/source controls
    # We include categorical Saffir-Simpson Category and Source as fixed-effect style controls
    formula_deaths = (
        'LogDeaths ~ MasFem_z + IsFemaleName + MaxWind + MinPressure + YearsSince '
        '+ C(Category) + C(Source) + C(Decade)'
    )

    # Fit with HC3 robust standard errors
    mod_deaths = smf.ols(formula=formula_deaths, data=df).fit(cov_type='HC3')
    results['model_deaths'] = mod_deaths

    # Auxiliary / robustness model: log damage (2013) as outcome
    formula_damage = (
        'LogDamage2013 ~ MasFem_z + IsFemaleName + MaxWind + MinPressure + YearsSince '
        '+ C(Category) + C(Source) + C(Decade)'
    )
    mod_damage = smf.ols(formula=formula_damage, data=df).fit(cov_type='HC3')
    results['model_damage'] = mod_damage

    # Additional robustness: use MTurk rating instead of original MasFem (if available)
    if 'MTurk_MasFem' in df.columns:
        df_copy = df.copy()
        if df_copy['MTurk_MasFem'].notna().any():
            mtmean = df_copy['MTurk_MasFem'].mean()
            mtstd = df_copy['MTurk_MasFem'].std(ddof=0)
            df_copy['MTurk_MasFem_z'] = (df_copy['MTurk_MasFem'] - mtmean) / (mtstd if (mtstd != 0 and not pd.isna(mtstd)) else 1.0)
            formula_mturk = (
                'LogDeaths ~ MTurk_MasFem_z + IsFemaleName + MaxWind + MinPressure + YearsSince '
                '+ C(Category) + C(Source) + C(Decade)'
            )
            results['model_deaths_mturk'] = smf.ols(formula=formula_mturk, data=df_copy).fit(cov_type='HC3')

    return results