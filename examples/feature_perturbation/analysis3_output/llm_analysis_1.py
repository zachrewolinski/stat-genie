from typing import Any, List
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid mutating original
    df = df.copy()

    # Helper: try candidate columns and return a numeric Series (with original index) if any convertible values exist.
    def find_numeric_series(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
        for c in candidates:
            if c in df.columns:
                # Try numeric conversion
                s_num = pd.to_numeric(df[c], errors='coerce')
                if s_num.notna().any():
                    return s_num
        return pd.Series(dtype=float, index=df.index)

    # Helper: find a series that may be categorical (e.g., 'male'/'female') and convert to binary 0/1 where possible.
    def find_binary_female_series(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
        for c in candidates:
            if c in df.columns:
                s = df[c]
                # If numeric convertible and has non-null, use numeric
                s_num = pd.to_numeric(s, errors='coerce')
                if s_num.notna().any():
                    # If values are not strictly 0/1 but numeric, keep numeric (assume already coded)
                    return s_num
                # Otherwise try mapping common string labels
                s_str = s.astype(str).str.strip().str.lower()
                mapping = {'female': 1, 'f': 1, 'woman': 1, 'female_name': 1,
                           'male': 0, 'm': 0, 'man': 0}
                mapped = s_str.map(mapping)
                if mapped.notna().any():
                    return mapped
        return pd.Series(dtype=float, index=df.index)

    # Candidate source columns for each conceptual final variable.
    death_candidates = ['ndam15', 'deaths', 'fatalities', 'ndead', 'fatality']
    femininity_coder_candidates = [
        'Femininity_Coder', 'femininity_coder', 'masfem_coder', 'masfemcoder',
        'coder_fem', 'coder_rating', 'name', 'masfem'  # include masfem as potential source if it encodes a rating
    ]
    femininity_mturk_candidates = ['masfem_mturk', 'masfem-mturk', 'mturk_fem', 'femininity_mturk']
    binary_female_candidates = ['elapsedyrs', 'female', 'female_name', 'is_female', 'female_name_ind', 'sex', 'sex_f']
    maxwind_candidates = ['wind', 'max_wind', 'maxwind', 'wind_speed']
    minpressure_candidates = ['min', 'min_pressure', 'pressure_min', 'minpress']
    saffir_candidates = ['masfem', 'saffir_simpson', 'category', 'ss_category', 'saffir']
    year_candidates = ['alldeaths', 'year', 'yr']
    damage_candidates = ['ind', 'damage', 'adjusted_damage', 'adj_damage', 'economic_damage']

    # Create LogDeaths
    deaths_series = find_numeric_series(df, death_candidates)
    if not deaths_series.empty and deaths_series.notna().any():
        df['LogDeaths'] = np.log1p(deaths_series.astype(float))
    else:
        # If no numeric deaths found, create empty column of floats (will be dropped later if required)
        df['LogDeaths'] = pd.Series(dtype=float, index=df.index)

    # Femininity coder: try to find numeric rating. If candidate yields non-numeric (e.g., hurricane names), skip.
    fem_coder_series = find_numeric_series(df, femininity_coder_candidates)
    # If we didn't find numeric coder ratings but MTURK ratings exist, use MTURK as a fallback to populate Femininity_Coder
    fem_mturk_series = find_numeric_series(df, femininity_mturk_candidates)
    if fem_coder_series.notna().any():
        df['Femininity_Coder_c'] = fem_coder_series.astype(float) - fem_coder_series.astype(float).mean()
    else:
        # If no coder ratings but MTURK exists, use MTURK to fill coder variable as a fallback.
        if fem_mturk_series.notna().any():
            df['Femininity_Coder_c'] = fem_mturk_series.astype(float) - fem_mturk_series.astype(float).mean()
        else:
            # Create empty column
            df['Femininity_Coder_c'] = pd.Series(dtype=float, index=df.index)

    # MTURK femininity (optional)
    if fem_mturk_series.notna().any():
        df['Femininity_MTURK_c'] = fem_mturk_series.astype(float) - fem_mturk_series.astype(float).mean()
    else:
        # Do NOT create a column of all-NaNs if no MTURK ratings are present (follow original intent).
        # We'll simply not add Femininity_MTURK_c in that case.
        if 'Femininity_MTURK_c' in df.columns:
            df = df.drop(columns=['Femininity_MTURK_c'])

    # Binary female-name indicator
    binary_series = find_binary_female_series(df, binary_female_candidates)
    if not binary_series.empty and binary_series.notna().any():
        df['BinaryFemaleName'] = binary_series.astype(float)
    else:
        # Create empty column if no source found (will be handled downstream)
        df['BinaryFemaleName'] = pd.Series(dtype=float, index=df.index)

    # Intensity controls
    maxwind_series = find_numeric_series(df, maxwind_candidates)
    if not maxwind_series.empty and maxwind_series.notna().any():
        df['MaxWind'] = maxwind_series.astype(float)
    else:
        df['MaxWind'] = pd.Series(dtype=float, index=df.index)

    minpress_series = find_numeric_series(df, minpressure_candidates)
    if not minpress_series.empty and minpress_series.notna().any():
        df['MinPressure'] = minpress_series.astype(float)
    else:
        df['MinPressure'] = pd.Series(dtype=float, index=df.index)

    # Saffir-Simpson category proxy
    saffir_series = find_numeric_series(df, saffir_candidates)
    if not saffir_series.empty and saffir_series.notna().any():
        df['SaffirSimpson'] = saffir_series.astype(float)
    else:
        df['SaffirSimpson'] = pd.Series(dtype=float, index=df.index)

    # Year
    year_series = find_numeric_series(df, year_candidates)
    if not year_series.empty and year_series.notna().any():
        df['Year'] = year_series.astype(float)
    else:
        df['Year'] = pd.Series(dtype=float, index=df.index)

    # LogDamage
    damage_series = find_numeric_series(df, damage_candidates)
    if not damage_series.empty and damage_series.notna().any():
        df['LogDamage'] = np.log1p(damage_series.astype(float))
    else:
        df['LogDamage'] = pd.Series(dtype=float, index=df.index)

    # Now enforce that the FINAL required columns are present and non-missing.
    final_required = [
        'LogDeaths',
        'Femininity_Coder_c',
        'BinaryFemaleName',
        'MaxWind',
        'MinPressure',
        'SaffirSimpson',
        'Year',
        'LogDamage'
    ]
    # If any of these columns are entirely missing from the df at the moment (shouldn't be), create empty cols
    for col in final_required:
        if col not in df.columns:
            df[col] = pd.Series(dtype=float, index=df.index)

    # Drop rows with missing values in required columns (must have complete cases for modeling)
    df = df.dropna(subset=final_required)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Ensure required final columns exist
    required = [
        'LogDeaths',
        'Femininity_Coder_c',
        'BinaryFemaleName',
        'MaxWind',
        'MinPressure',
        'SaffirSimpson',
        'Year',
        'LogDamage'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required column(s) for modeling missing from dataframe: {missing}")

    # Ensure there are observations with non-missing values for the primary specification
    cols_spec1 = required.copy()
    df1 = df.dropna(subset=cols_spec1)
    if df1.empty:
        raise ValueError("No observations available for modeling after transform (all rows missing required values).")

    # Specification 1 (primary): test relationship between coder-rated femininity and log deaths,
    # controlling for storm intensity, year, and economic damage. Use robust (HC3) SEs.
    formula1 = 'LogDeaths ~ Femininity_Coder_c + BinaryFemaleName + MaxWind + MinPressure + SaffirSimpson + Year + LogDamage'
    model1 = smf.ols(formula1, data=df1).fit(cov_type='HC3')

    results = {'model_coder': model1}

    # Specification 2 (robustness): replace coder-rated femininity with MTurk-rated femininity if available
    if 'Femininity_MTURK_c' in df.columns and df['Femininity_MTURK_c'].notnull().any():
        cols_spec2 = ['LogDeaths', 'Femininity_MTURK_c', 'BinaryFemaleName', 'MaxWind', 'MinPressure', 'SaffirSimpson', 'Year', 'LogDamage']
        df2 = df.dropna(subset=cols_spec2)
        if not df2.empty:
            formula2 = 'LogDeaths ~ Femininity_MTURK_c + BinaryFemaleName + MaxWind + MinPressure + SaffirSimpson + Year + LogDamage'
            model2 = smf.ols(formula2, data=df2).fit(cov_type='HC3')
            results['model_mturk'] = model2
        # If df2 is empty, skip adding model_mturk (no usable rows)

    return results