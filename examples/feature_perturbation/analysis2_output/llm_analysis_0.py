from typing import Any
import re
import numpy as np
import pandas as pd

# Helper to canonicalize column names for robust matching
def _canonical(col: str) -> str:
    return re.sub(r'\W+', '', str(col)).lower()


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a raw hurricane dataset into the final dataframe required by the model.

    The final dataframe must contain the exact column names used by the analysis:
    ['MasFem_z', 'NameGender', 'Fatalities', 'MaxWind', 'MinPressure', 'Category', 'Year_c', 'MTurkMasFem_z', ...]
    This function is robust to minor variations in incoming column names such as 'feature4', 'Feature 4', etc.
    """
    df = df.copy()

    # Mapping from possible original feature names to the required target names
    desired_mapping = {
        'feature1': 'StormID',
        'feature2': 'Year',
        'feature3': 'Name',
        'feature4': 'MasFem',
        'feature5': 'MinPressure',
        'feature6': 'NameGender',
        'feature7': 'Category',
        'feature8': 'Fatalities',
        'feature9': 'Damage_norm2013',
        'feature10': 'YearsSince',
        'feature11': 'Source',
        'feature12': 'MTurkMasFem',
        'feature13': 'MaxWind',
        'feature14': 'Damage_norm2015'
    }

    # Build a lookup from canonicalized column name to actual column name present in df
    present_cols_canon = { _canonical(c): c for c in df.columns }

    # For each desired mapping, if either the original key or the target name (already present) matches a column, rename it to the target
    rename_map = {}
    for orig, target in desired_mapping.items():
        orig_c = _canonical(orig)
        # If a column already has the correct target name, no rename needed
        if target in df.columns:
            continue
        # If a column matching the original exists, rename it
        if orig_c in present_cols_canon:
            rename_map[present_cols_canon[orig_c]] = target
        else:
            # As a fallback, try to find any column whose canonical matches the canonical of the original
            for c in df.columns:
                if _canonical(c) == orig_c:
                    rename_map[c] = target
                    break

    if rename_map:
        df = df.rename(columns=rename_map)

    # Now coerce key columns to numeric where appropriate (safely)
    numeric_cols = ['MasFem', 'MinPressure', 'NameGender', 'Category', 'Fatalities',
                    'Damage_norm2013', 'YearsSince', 'MTurkMasFem', 'MaxWind',
                    'Damage_norm2015', 'Year']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing essential variables needed for modeling (only for columns that exist)
    required = ['MasFem', 'Fatalities', 'MaxWind', 'MinPressure', 'Category', 'Year']
    present_required = [c for c in required if c in df.columns]
    if present_required:
        df = df.dropna(subset=present_required)

    # Standardize continuous femininity measures for interpretation
    # Use population std (ddof=0) for stable z-score
    if 'MasFem' in df.columns:
        mas_mean = df['MasFem'].mean()
        mas_std = df['MasFem'].std(ddof=0)
        if pd.isna(mas_std) or mas_std == 0:
            mas_std = 1.0
        df['MasFem_z'] = (df['MasFem'] - mas_mean) / mas_std
    else:
        # Ensure the required final column exists even if input was missing
        df['MasFem_z'] = np.nan

    if 'MTurkMasFem' in df.columns:
        mt_mean = df['MTurkMasFem'].mean()
        mt_std = df['MTurkMasFem'].std(ddof=0)
        if pd.isna(mt_std) or mt_std == 0:
            mt_std = 1.0
        df['MTurkMasFem_z'] = (df['MTurkMasFem'] - mt_mean) / mt_std
    else:
        df['MTurkMasFem_z'] = np.nan

    # Ensure binary gender indicator is integer 0/1 where possible
    if 'NameGender' in df.columns:
        # Coerce to numeric, fill missing with 0 (conservative default) and round
        df['NameGender'] = pd.to_numeric(df['NameGender'], errors='coerce').fillna(0).round().astype(int)
    else:
        # Create the column if missing (filled with 0s)
        df['NameGender'] = 0

    # Create a centered year covariate to aid interpretation and reduce collinearity
    if 'Year' in df.columns:
        df['Year_c'] = df['Year'] - df['Year'].mean()
    else:
        df['Year_c'] = np.nan

    # Ensure Category is treated as integer-coded categorical where possible
    if 'Category' in df.columns:
        # After dropping rows missing Category (above) this should be safe; fill remaining NaNs with 0 then cast
        df['Category'] = pd.to_numeric(df['Category'], errors='coerce').fillna(0).astype(int)
    else:
        df['Category'] = 0

    # Final check: Fatalities must be non-negative integer counts
    if 'Fatalities' in df.columns:
        df['Fatalities'] = df['Fatalities'].fillna(0).astype(int)
    else:
        # If missing, create a column of zeros (no fatalities information)
        df['Fatalities'] = 0

    # Return transformed dataframe containing all columns used in modeling (plus useful originals)
    keep_cols = ['StormID', 'Year', 'Year_c', 'Name', 'MasFem', 'MasFem_z', 'MTurkMasFem', 'MTurkMasFem_z',
                 'NameGender', 'Category', 'MaxWind', 'MinPressure', 'Fatalities',
                 'Damage_norm2013', 'Damage_norm2015', 'Source', 'YearsSince']
    cols_present = [c for c in keep_cols if c in df.columns]
    # Ensure all conceptual-final columns exist in returned DF (even if they were not in original)
    for required_final in ['MasFem_z', 'NameGender', 'Fatalities', 'MaxWind', 'MinPressure', 'Category', 'Year_c', 'MTurkMasFem_z']:
        if required_final not in cols_present:
            # create with NaNs or sensible defaults
            if required_final in ['NameGender', 'Fatalities', 'Category']:
                df[required_final] = 0
            else:
                df[required_final] = np.nan
            cols_present.append(required_final)

    # Ensure deterministic column order for returned dataframe
    # Place the conceptual variables first in a sensible order, then any other kept columns
    final_order = ['MasFem_z', 'NameGender', 'Fatalities', 'MaxWind', 'MinPressure', 'Category', 'Year_c', 'MTurkMasFem_z']
    # Add any other requested keep_cols that exist and are not already in final_order
    for c in cols_present:
        if c not in final_order:
            final_order.append(c)
    # Filter to only columns actually present
    final_order = [c for c in final_order if c in df.columns]

    return df[final_order]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Runs several regression specifications appropriate for a count outcome (Fatalities).
    Primary specification: negative binomial GLM with continuous MasFem_z.
    Secondary specification: negative binomial GLM with binary NameGender.
    Returns a dict of fitted results objects for further inspection.
    If there is insufficient data for a given specification (no rows with all required variables non-missing),
    that specification is skipped.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    results = {}

    # Define required column sets for each specification
    req_mas = ['Fatalities', 'MasFem_z', 'MaxWind', 'MinPressure', 'Category', 'Year_c']
    req_bin = ['Fatalities', 'NameGender', 'MaxWind', 'MinPressure', 'Category', 'Year_c']
    req_mturk = ['Fatalities', 'MTurkMasFem_z', 'MaxWind', 'MinPressure', 'Category', 'Year_c']

    # Helper to check if specification has any usable rows
    def has_data(required_cols):
        # All required columns must exist
        for col in required_cols:
            if col not in df.columns:
                return False
        # At least one row must have non-missing values for all required columns
        subset = df[required_cols].dropna()
        return subset.shape[0] > 0

    # Helper to fit safely
    def fit_spec(formula, required_cols, result_key):
        if not has_data(required_cols):
            return  # skip if insufficient data
        try:
            nb_res = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial()).fit()
            results[result_key] = nb_res
        except Exception:
            # Fallback to Poisson with robust SEs
            pois_res = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()
            pois_robust = pois_res.get_robustcov_results(cov_type='HC0')
            results[f"{result_key}_poisson_robust"] = pois_robust

    # Formula using standardized femininity score
    formula_mas = 'Fatalities ~ MasFem_z + MaxWind + MinPressure + C(Category) + Year_c'
    fit_spec(formula_mas, req_mas, 'masfem_nb')

    # Binary name-gender specification
    formula_bin = 'Fatalities ~ NameGender + MaxWind + MinPressure + C(Category) + Year_c'
    fit_spec(formula_bin, req_bin, 'genderbin_nb')

    # Robustness: use MTurk rating if available
    formula_mturk = 'Fatalities ~ MTurkMasFem_z + MaxWind + MinPressure + C(Category) + Year_c'
    fit_spec(formula_mturk, req_mturk, 'mturk_masfem_nb')

    return results