from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw Simonsohn hurricane dataframe into a modeling-ready dataframe.

    Expected input columns (per provided schema):
      - feature2: Year
      - feature3: Name
      - feature4: masculinity-femininity index (higher = more feminine)
      - feature5: Minimum central pressure at landfall
      - feature6: Binary gender label of name (0=male, 1=female)
      - feature7: Saffir-Simpson category
      - feature8: Fatalities (raw count)
      - feature10: Years since event (or similar time metric)
      - feature11: Source (categorical)
      - feature13: Maximum wind speed
      - feature14: Damage normalized to 2015 dollars

    Returns a dataframe with the exact columns referenced in the conceptual model and modeling code.
    """
    df = df.copy()

    # Rename raw feature columns to meaningful column names used in modeling
    rename_map = {
        'feature2': 'Year',
        'feature3': 'Name',
        'feature4': 'MasFem',
        'feature5': 'MinPressure',
        'feature6': 'Female',
        'feature7': 'SSCategory',
        'feature8': 'Fatalities',
        'feature10': 'YearsSince',
        'feature11': 'Source',
        'feature13': 'MaxWind',
        'feature14': 'Damage2015'
    }
    df = df.rename(columns=rename_map)

    # Ensure existence of the conceptual columns (create with NaN or sensible defaults if missing)
    required_conceptual_cols = [
        'MasFem', 'Fatalities', 'MinPressure', 'MaxWind',
        'Damage2015', 'YearsSince', 'Year', 'Female',
        'SSCategory', 'Source', 'Name'
    ]
    for col in required_conceptual_cols:
        if col not in df.columns:
            if col in ['SSCategory', 'Source', 'Name']:
                # Categorical/string defaults
                df[col] = pd.Series([pd.NA] * len(df))
            else:
                # Numeric defaults
                df[col] = np.nan

    # Keep only rows with the key variables non-missing; these are required for main analyses
    required_cols = ['MasFem', 'Fatalities', 'MinPressure', 'MaxWind']
    df = df.dropna(subset=required_cols)

    # Ensure numeric dtype for numeric columns
    numeric_cols = ['MasFem', 'MinPressure', 'MaxWind', 'Fatalities', 'Damage2015', 'YearsSince', 'Year']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # If numeric conversion introduced NaNs in required columns, drop those rows
    df = df.dropna(subset=['MasFem', 'Fatalities', 'MinPressure', 'MaxWind'])

    # Create dependent variable transformations
    df['LogFatalities'] = np.log1p(df['Fatalities'].astype(float))

    # Damage: fill missing with zero when appropriate (no reported damage) then log-transform
    if 'Damage2015' in df.columns:
        df['Damage2015'] = df['Damage2015'].fillna(0).astype(float)
    else:
        df['Damage2015'] = 0.0
    df['LogDamage2015'] = np.log1p(df['Damage2015'])

    # Standardize the MasFem index (z-score) for easy interpretation of coefficients
    masfem_mean = df['MasFem'].mean()
    masfem_std = df['MasFem'].std(ddof=0)
    if np.isnan(masfem_mean) or (masfem_std == 0 or np.isnan(masfem_std)):
        # If degenerate, produce zeros to avoid NaNs
        df['MasFem_z'] = 0.0
    else:
        df['MasFem_z'] = (df['MasFem'] - masfem_mean) / masfem_std

    # Ensure binary Female is integer (0/1) where possible
    if 'Female' in df.columns:
        # coerce to numeric then to 0/1
        df['Female'] = pd.to_numeric(df['Female'], errors='coerce')
        # If values are not strictly 0/1, force them into 0/1 by thresholding at 0.5
        df['Female'] = df['Female'].fillna(0).apply(lambda x: 1 if x >= 0.5 else 0).astype(int)
    else:
        df['Female'] = 0

    # Categorical controls
    if 'SSCategory' in df.columns:
        # Replace missing category with 'Unknown' for stability
        df['SSCategory'] = df['SSCategory'].fillna('Unknown').astype('category')
    else:
        df['SSCategory'] = pd.Series(['Unknown'] * len(df), dtype='category')

    if 'Source' in df.columns:
        df['Source'] = df['Source'].fillna('Unknown').astype('category')
    else:
        df['Source'] = pd.Series(['Unknown'] * len(df), dtype='category')

    # Keep a compact set of columns required for modeling and traceability
    keep_cols = [
        'Name', 'Year', 'YearsSince',
        'MasFem', 'MasFem_z', 'Female',
        'MinPressure', 'MaxWind', 'SSCategory',
        'Fatalities', 'LogFatalities',
        'Damage2015', 'LogDamage2015',
        'Source'
    ]
    # Subset accordingly (all should exist due to earlier creation)
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Final: drop any rows with missing DV or key IV after transformation
    final_required = ['LogFatalities', 'MasFem_z', 'MinPressure', 'MaxWind']
    df = df.dropna(subset=final_required)

    # Ensure types: Year and YearsSince numeric where possible
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    if 'YearsSince' in df.columns:
        df['YearsSince'] = pd.to_numeric(df['YearsSince'], errors='coerce')

    # Ensure categorical columns are category dtype
    if 'SSCategory' in df.columns:
        df['SSCategory'] = df['SSCategory'].astype('category')
    if 'Source' in df.columns:
        df['Source'] = df['Source'].astype('category')

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit primary and robustness models testing whether more-feminine hurricane names are associated
    with fewer precautionary measures (proxied here by higher fatalities). Because precautionary
    behavior is not directly observed in the dataset, we use fatalities as the primary outcome
    (both log-transformed for OLS and raw counts for a negative-binomial model).

    Returns a dict with fitted model results objects for further inspection.
    """
    # Copy to avoid modifying input
    df = df.copy()

    # Helper to extract variable names from RHS term strings (handles C(var) notation)
    def _varname_from_term(term: str) -> str:
        t = term.strip()
        if t.startswith('C(') and t.endswith(')'):
            return t[2:-1]
        return t

    # Build RHS of formula conditionally including categorical controls only when they vary.
    base_rhs_terms = [
        'MasFem_z', 'MinPressure', 'MaxWind', 'LogDamage2015', 'YearsSince'
    ]
    rhs_terms = base_rhs_terms.copy()

    # Include categorical controls only if they have more than one non-missing level.
    # If they do not vary (0 or 1 level), including them in the formula will cause patsy errors.
    for cat in ['SSCategory', 'Source']:
        if cat in df.columns:
            n_levels = int(df[cat].nunique(dropna=True))
            if n_levels > 1:
                rhs_terms.append(f'C({cat})')

    # Before attempting to fit, ensure there are observations with non-missing values
    # for the outcome and all RHS variables. This avoids statsmodels raising errors on empty data.
    rhs_varnames = [_varname_from_term(t) for t in rhs_terms]
    required_for_ols = set(rhs_varnames + ['LogFatalities'])
    # Keep only those that actually exist in the dataframe
    required_for_ols = [v for v in required_for_ols if v in df.columns]
    df_ols_check = df[required_for_ols].dropna()
    # Prepare placeholders for results that might fail to fit
    ols_res = None
    ols_res_raw = None

    if df_ols_check.shape[0] > 0:
        rhs = ' + '.join(rhs_terms)
        formula = f'LogFatalities ~ {rhs}'
        # Primary OLS on log fatalities with robust SEs
        try:
            ols_res_raw = smf.ols(formula, data=df).fit()
            ols_res = ols_res_raw.get_robustcov_results(cov_type='HC3')
        except Exception:
            ols_res_raw = None
            ols_res = None
    else:
        # Not enough data to fit OLS; leave results as None
        ols_res_raw = None
        ols_res = None

    # Robustness 1: use the binary Female label instead of continuous index
    rhs_terms_bin = ['Female', 'MinPressure', 'MaxWind', 'LogDamage2015', 'YearsSince']
    for cat in ['SSCategory', 'Source']:
        if cat in df.columns:
            n_levels = int(df[cat].nunique(dropna=True))
            if n_levels > 1:
                rhs_terms_bin.append(f'C({cat})')

    rhs_varnames_bin = [_varname_from_term(t) for t in rhs_terms_bin]
    required_for_ols_bin = set(rhs_varnames_bin + ['LogFatalities'])
    required_for_ols_bin = [v for v in required_for_ols_bin if v in df.columns]
    df_ols_bin_check = df[required_for_ols_bin].dropna()
    ols_bin_res = None
    ols_bin_res_raw = None

    if df_ols_bin_check.shape[0] > 0:
        rhs_bin = ' + '.join(rhs_terms_bin)
        formula_bin = f'LogFatalities ~ {rhs_bin}'
        try:
            ols_bin_res_raw = smf.ols(formula_bin, data=df).fit()
            ols_bin_res = ols_bin_res_raw.get_robustcov_results(cov_type='HC3')
        except Exception:
            ols_bin_res_raw = None
            ols_bin_res = None
    else:
        ols_bin_res_raw = None
        ols_bin_res = None

    # Robustness 2: count model (Negative Binomial) on raw Fatalities (counts)
    # Build design matrix with dummies for categorical vars that vary
    cat_cols = []
    for cat in ['SSCategory', 'Source']:
        if cat in df.columns:
            if int(df[cat].nunique(dropna=True)) > 1:
                cat_cols.append(cat)

    nb_X_cols = ['MasFem_z', 'MinPressure', 'MaxWind', 'LogDamage2015', 'YearsSince']
    # Ensure expected columns exist; transform should have created them, but validate
    nb_X_cols_present = [c for c in nb_X_cols if c in df.columns]
    X = df[nb_X_cols_present].copy()

    if cat_cols:
        # get_dummies will simply not add columns for any categorical that has a single level,
        # but we've already filtered to only include those with >1 level
        X = pd.concat([X, pd.get_dummies(df[cat_cols], drop_first=True)], axis=1)

    X = sm.add_constant(X, has_constant='add')

    # Ensure Fatalities is integer counts and drop rows with missing values in X or y
    if 'Fatalities' not in df.columns:
        # Cannot fit count model without Fatalities
        nb_model = None
    else:
        y = df['Fatalities']
        model_df = X.join(y).dropna()
        if model_df.shape[0] == 0 or model_df.shape[1] <= 1:
            # No usable rows or no predictors (only constant and no data) -> cannot fit
            nb_model = None
        else:
            y_clean = model_df['Fatalities'].astype(int)
            X_clean = model_df.drop(columns=['Fatalities'])
            # Fit a Negative Binomial GLM to account for count nature and overdispersion
            try:
                nb_fit_raw = sm.GLM(y_clean, X_clean, family=sm.families.NegativeBinomial()).fit()
                nb_model = nb_fit_raw
            except Exception:
                # If GLM NB fails to converge, fall back to Poisson and attach robust covariances
                try:
                    poisson_fit_raw = sm.GLM(y_clean, X_clean, family=sm.families.Poisson()).fit()
                    nb_model = poisson_fit_raw.get_robustcov_results(cov_type='HC3')
                except Exception:
                    nb_model = None

    # Return a dict of results for inspection
    results = {
        'ols_logfatalities_masfem': ols_res,
        'ols_logfatalities_femalebin': ols_bin_res,
        'count_model_fatalities_nb_or_poisson': nb_model
    }
    return results