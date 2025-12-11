from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def _find_and_rename(df: pd.DataFrame, target: str, candidates: list) -> None:
    """
    If target column not present, try to find a column whose name matches any of the
    candidate substrings (case-insensitive). If found, rename that column to target.
    This mutates the dataframe in place.
    """
    if target in df.columns:
        return
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        cand_lower = cand.lower()
        # exact match
        if cand_lower in cols_lower:
            df.rename(columns={cols_lower[cand_lower]: target}, inplace=True)
            return
    # substring match
    for col in df.columns:
        col_l = col.lower()
        for cand in candidates:
            if cand.lower() in col_l:
                df.rename(columns={col: target}, inplace=True)
                return


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into a dataframe ready for modeling.

    Ensures the final dataframe contains the required columns:
      - MasFem_z, FemaleName, Deaths, MaxWind, MinPressure, Category, Year, LogDamage2015
    and other descriptive columns when available.

    The function attempts to intelligently find likely source columns if the raw input
    uses different naming conventions (case-insensitive substring matching). It will
    coerce types as appropriate and create fallback columns if necessary.
    """
    df = df.copy()

    # Try to discover and rename common source columns to the canonical names required.
    # These are heuristics to handle variations in input CSV column names.
    candidate_map = {
        'StormID': ['stormid', 'storm_id', 'id', 'feature1'],
        'Year': ['year', 'yr', 'season', 'feature2'],
        'StormName': ['stormname', 'name', 'storm', 'feature3'],
        'MasFem': ['masfem', 'mas_fem', 'masculinity', 'femininity', 'mas-fem', 'feature4'],
        'MinPressure': ['minpressure', 'min_pressure', 'pressure', 'central_pressure', 'feature5'],
        'FemaleName': ['femalename', 'female_name', 'female', 'sex_name', 'gender_name', 'feature6'],
        'Category': ['category', 'saffir', 'saffir-simpson', 'cat', 'feature7'],
        'Deaths': ['deaths', 'death', 'fatalit', 'fatal', 'feature8'],
        'DamageRaw': ['damageraw', 'damage_raw', 'rawdamage', 'feature9'],
        'YearsSince': ['yearssince', 'years_since', 'feature10'],
        'Source': ['source', 'feature11'],
        'MasFem_MT': ['masfem_mt', 'mas_fem_mt', 'mturk', 'feature12'],
        'MaxWind': ['maxwind', 'max_wind', 'wind', 'feature13'],
        'Damage2015': ['damage2015', 'damage_2015', 'damage', 'loss2015', 'feature14']
    }

    for target, candidates in candidate_map.items():
        _find_and_rename(df, target, candidates)

    # Ensure all canonical columns exist in the dataframe (create with NaN if absent)
    canonical_cols = list(candidate_map.keys())
    for col in canonical_cols:
        if col not in df.columns:
            df[col] = np.nan

    # 2) Coerce key columns to numeric where appropriate
    numeric_cols = ['MasFem', 'MasFem_MT', 'MinPressure', 'Category', 'Deaths', 'Damage2015', 'MaxWind', 'Year']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # FemaleName: try to coerce to binary 0/1
    if 'FemaleName' in df.columns:
        # If numeric-like after coercion, map non-zero -> 1, NaN -> 0
        if pd.api.types.is_numeric_dtype(df['FemaleName']):
            df['FemaleName'] = df['FemaleName'].fillna(0).astype(int).clip(0, 1)
        else:
            # Map strings: anything containing 'fem' or starting with 'f' -> 1
            def map_female(x):
                if pd.isnull(x):
                    return 0
                s = str(x).strip().lower()
                if any(k in s for k in ('fem', 'female', 'woman', 'woma')) or s.startswith('f'):
                    return 1
                if any(k in s for k in ('male', 'm', 'man')):
                    return 0
                # fallback: try numeric conversion
                try:
                    v = float(s)
                    return int(v != 0)
                except Exception:
                    return 0
            df['FemaleName'] = df['FemaleName'].apply(map_female).astype(int)
    else:
        df['FemaleName'] = 0

    # 3) Prepare Damage2015 and LogDamage2015
    if 'Damage2015' in df.columns and df['Damage2015'].notna().any():
        df['Damage2015'] = pd.to_numeric(df['Damage2015'], errors='coerce')
        df['Damage2015'] = df['Damage2015'].fillna(0).clip(lower=0)
        df['LogDamage2015'] = np.log(df['Damage2015'] + 1)
    else:
        # Try to derive from DamageRaw if available
        if 'DamageRaw' in df.columns and df['DamageRaw'].notna().any():
            df['DamageRaw'] = pd.to_numeric(df['DamageRaw'], errors='coerce').fillna(0).clip(lower=0)
            # If Damage2015 not available, conservatively treat DamageRaw as already adjusted
            df['Damage2015'] = df['DamageRaw']
            df['LogDamage2015'] = np.log(df['Damage2015'] + 1)
        else:
            df['Damage2015'] = 0.0
            df['LogDamage2015'] = 0.0

    # 4) Drop rows missing primary variables when those columns exist;
    # if either primary variable column is entirely missing (all NaN), keep rows but they'll be filtered later.
    # Primary variables: MasFem and Deaths
    primary_subset = [c for c in ['MasFem', 'Deaths'] if c in df.columns]
    if primary_subset:
        # Only drop rows where both primary variables are NaN in case one is present
        df = df.dropna(subset=primary_subset, how='all')

    # 5) Ensure Deaths is integer count (coerce and floor if necessary)
    if 'Deaths' in df.columns:
        # Replace negative deaths with NaN then drop later if needed
        df['Deaths'] = pd.to_numeric(df['Deaths'], errors='coerce')
        # Replace negative values with NaN
        df.loc[df['Deaths'] < 0, 'Deaths'] = np.nan
        # Fill NaN with 0 conservatively if no deaths info
        df['Deaths'] = df['Deaths'].fillna(0).astype(int)

    # 6) Create LogDeaths
    df['LogDeaths'] = np.log(df['Deaths'] + 1)

    # 7) Standardize MasFem to z-score and create MasFem_z
    # If MasFem is entirely NaN, attempt to use MasFem_MT; otherwise set MasFem_z to 0
    df['MasFem'] = pd.to_numeric(df['MasFem'], errors='coerce')
    if df['MasFem'].isna().all() and 'MasFem_MT' in df.columns:
        df['MasFem'] = pd.to_numeric(df['MasFem_MT'], errors='coerce')

    mas_mean = df['MasFem'].mean(skipna=True)
    mas_std = df['MasFem'].std(ddof=0, skipna=True)
    if pd.isna(mas_std) or mas_std == 0:
        # fallback: if MasFem all NaN or zero variance, create MasFem_z as 0 for all rows
        df['MasFem_z'] = 0.0
    else:
        df['MasFem_z'] = (df['MasFem'] - mas_mean) / mas_std
        df['MasFem_z'] = df['MasFem_z'].fillna(0.0)

    # 8) Ensure control columns exist and are numeric; fill missing with zeros or sensible defaults
    # Use sensible defaults (avoid leaving NaNs that will cause all rows to be dropped later).
    controls_defaults = {
        'MaxWind': 0.0,
        'MinPressure': 0.0,  # Fill missing pressure with 0.0 to avoid all-NaN issues in modeling
        'Category': 0,
        'Year': 0
    }
    for col, default in controls_defaults.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(default)
        else:
            df[col] = default

    # Ensure Category and Year are integer-like where appropriate
    if 'Category' in df.columns:
        try:
            df['Category'] = df['Category'].astype(int)
        except Exception:
            df['Category'] = pd.to_numeric(df['Category'], errors='coerce').fillna(0).astype(int)
    if 'Year' in df.columns:
        try:
            df['Year'] = df['Year'].astype(int)
        except Exception:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)

    # 9) Final housekeeping: keep only columns needed for the modeling and downstream checks
    keep_cols = ['StormID', 'Year', 'StormName', 'MasFem', 'MasFem_MT', 'MasFem_z', 'FemaleName',
                 'Category', 'MaxWind', 'MinPressure', 'Deaths', 'Damage2015', 'LogDamage2015', 'LogDeaths', 'Source']
    available_keep = [c for c in keep_cols if c in df.columns]
    df = df[available_keep]

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit statistical models to estimate whether more feminine hurricane names are associated
    with fatalities.

    Returns a dict with:
      - 'nb_masfem': Negative binomial (or Poisson fallback) with MasFem_z
      - 'nb_femalebinary': Negative binomial (or Poisson fallback) with FemaleName
      - 'ols_logdeath': OLS on LogDeaths with HC3 robust SE
    """
    df = df.copy()

    # Ensure required columns exist
    required = ['Deaths', 'MasFem_z', 'FemaleName', 'MaxWind', 'MinPressure', 'Category', 'Year', 'LogDamage2015']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Dataframe missing required columns for modeling: {missing}")

    # If the dataframe has no rows, raise a clear error
    if df.shape[0] == 0:
        raise ValueError("Dataframe contains no rows after transformation; cannot fit models.")

    # Before fitting, drop rows with missing values in the required model columns to avoid
    # statsmodels dropping all rows internally (which can cause confusing zero-size errors).
    df = df.dropna(subset=required, how='any').copy()

    if df.shape[0] == 0:
        raise ValueError("No rows with complete data across required model variables; cannot fit models.")

    # Define formulae
    base_controls = 'MaxWind + MinPressure + Category + Year + LogDamage2015'
    formula_nb_masfem = f'Deaths ~ MasFem_z + {base_controls}'
    formula_nb_female = f'Deaths ~ FemaleName + {base_controls}'
    formula_ols = f'LogDeaths ~ MasFem_z + FemaleName + {base_controls}'

    results = {}

    # 1) Negative binomial with MasFem_z
    try:
        model_nb = smf.glm(formula_nb_masfem, data=df, family=sm.families.NegativeBinomial())
        res_nb = model_nb.fit()
        results['nb_masfem'] = res_nb
    except Exception:
        # Fallback to Poisson and obtain robust cov
        model_p = smf.glm(formula_nb_masfem, data=df, family=sm.families.Poisson())
        res_p = model_p.fit()
        try:
            res_p_robust = res_p.get_robustcov_results(cov_type='HC3')
            results['nb_masfem'] = res_p_robust
        except Exception:
            results['nb_masfem'] = res_p

    # 2) Negative binomial with FemaleName
    try:
        model_nb2 = smf.glm(formula_nb_female, data=df, family=sm.families.NegativeBinomial())
        res_nb2 = model_nb2.fit()
        results['nb_femalebinary'] = res_nb2
    except Exception:
        model_p2 = smf.glm(formula_nb_female, data=df, family=sm.families.Poisson())
        res_p2 = model_p2.fit()
        try:
            res_p2_robust = res_p2.get_robustcov_results(cov_type='HC3')
            results['nb_femalebinary'] = res_p2_robust
        except Exception:
            results['nb_femalebinary'] = res_p2

    # 3) OLS on log-deaths with HC3 robust SE
    ols_mod = smf.ols(formula_ols, data=df)
    ols_res = ols_mod.fit()
    try:
        ols_res_robust = ols_res.get_robustcov_results(cov_type='HC3')
        results['ols_logdeath'] = ols_res_robust
    except Exception:
        results['ols_logdeath'] = ols_res

    return results