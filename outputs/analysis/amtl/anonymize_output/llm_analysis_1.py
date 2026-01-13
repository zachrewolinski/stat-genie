from typing import Any
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Example top-level read (kept for compatibility; transform accepts a dataframe argument)
# Adjust path as needed when running in a different environment.
try:
    df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/amtl/anonymize_output/amtl.csv')
except Exception:
    df = pd.DataFrame()


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw input dataframe (with original columns feature1..feature9 or possibly already-renamed
    columns) into the final analytic dataframe.

    Produces columns used in the model (exact required final column names are preserved):
      - MissingCount: number of missing teeth of given class (integer, clipped to socket count)
      - SocketCount: number of observable sockets (integer, >0)
      - Genus: cleaned categorical genus (e.g., 'Homo sapiens', 'Pan', 'Pongo', 'Papio')
      - ToothClass: cleaned tooth class categorical ('Anterior','Posterior','Premolar')
      - AgeAtDeath: original estimated age
      - AgeUncertainty: uncertainty in age estimate
      - SexEstimate: original numeric sex estimate
      - SexMale: binary sex indicator (1 if SexEstimate >= 0.5 else 0)
      - Age_std: standardized age (mean 0, sd 1)
      - Region: cleaned region
      - SpecimenID: identifier (kept for reference)
    """
    # Work on a copy
    df = df.copy()

    # Rename incoming feature columns to meaningful names used in modeling when present
    rename_map = {
        'feature1': 'ToothClass',
        'feature2': 'SpecimenID',
        'feature3': 'MissingCount',
        'feature4': 'SocketCount',
        'feature5': 'AgeAtDeath',
        'feature6': 'AgeUncertainty',
        'feature7': 'SexEstimate',
        'feature8': 'Genus',
        'feature9': 'Region'
    }
    # Only rename those source columns that actually exist in the incoming dataframe
    rename_dict = {src: dst for src, dst in rename_map.items() if src in df.columns}
    if rename_dict:
        df = df.rename(columns=rename_dict)

    # Ensure all final required columns exist in the dataframe (create as NaN if missing so later coercion can operate)
    final_expected_cols = [
        'SpecimenID', 'MissingCount', 'SocketCount', 'Genus', 'ToothClass',
        'AgeAtDeath', 'AgeUncertainty', 'SexEstimate', 'Region'
    ]
    for col in final_expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Coerce numeric columns first (safe coercion to numeric)
    df['SocketCount'] = pd.to_numeric(df['SocketCount'], errors='coerce')
    df['MissingCount'] = pd.to_numeric(df['MissingCount'], errors='coerce')
    df['AgeAtDeath'] = pd.to_numeric(df['AgeAtDeath'], errors='coerce')
    df['AgeUncertainty'] = pd.to_numeric(df['AgeUncertainty'], errors='coerce')
    df['SexEstimate'] = pd.to_numeric(df['SexEstimate'], errors='coerce')

    # Clean categorical text columns and coerce to pandas string dtype while preserving NA
    def clean_text_series(s: pd.Series) -> pd.Series:
        s = s.copy()
        # Convert non-null values to string and strip; leave NA as-is
        notna_mask = s.notna()
        if notna_mask.any():
            s.loc[notna_mask] = s.loc[notna_mask].astype(str).str.strip()
        # Replace common placeholders that indicate missingness with actual NA
        s = s.replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA})
        # Ensure pandas string dtype so .str accessor works even if all values are NA
        try:
            s = s.astype("string")
        except Exception:
            # Fallback: return as-is if astype fails for unexpected reasons
            pass
        return s

    df['Genus'] = clean_text_series(df['Genus'])
    df['ToothClass'] = clean_text_series(df['ToothClass'])
    df['Region'] = clean_text_series(df['Region'])

    # Normalize common genus variants
    # Ensure we operate on string-safe series (some entries may be <NA>)
    df['Genus'] = df['Genus'].replace({
        'Homo': 'Homo sapiens',
        'homo sapiens': 'Homo sapiens',
        'Homo sapiens': 'Homo sapiens'
    })

    # Standardize tooth class labels (map several possible variants to canonical ones)
    # Use .str methods on pandas StringDtype; this is safe even when series is all NA
    if df['ToothClass'].dtype == "string" or df['ToothClass'].dtype == object:
        # Capitalize entries where present
        try:
            df['ToothClass'] = df['ToothClass'].str.capitalize()
        except Exception:
            # In rare cases where .str isn't applicable, convert to string safely first
            df['ToothClass'] = df['ToothClass'].astype("string")
            df['ToothClass'] = df['ToothClass'].str.capitalize()
    else:
        df['ToothClass'] = df['ToothClass'].astype("string")
        df['ToothClass'] = df['ToothClass'].str.capitalize()

    df['ToothClass'] = df['ToothClass'].replace({
        'Anterior': 'Anterior',
        'Posterior': 'Posterior',
        'Premolar': 'Premolar',
        'Premolar(s)': 'Premolar'
    })

    # SexEstimate: numeric proportion or score; produce a binary SexMale indicator
    df['SexMale'] = (df['SexEstimate'] >= 0.5).astype(float)  # keep float for now; will cast later

    # After coercion and cleaning, drop rows missing critical variables necessary for analysis
    required_cols = ['MissingCount', 'SocketCount', 'Genus', 'ToothClass', 'AgeAtDeath', 'SexEstimate']
    # If dataframe is empty (no rows), just ensure columns exist and return empty DF with expected columns
    if df.empty:
        # Create empty DataFrame with the final schema to preserve contract
        keep_cols = [
            'SpecimenID', 'MissingCount', 'SocketCount', 'PropMissing', 'Genus', 'ToothClass',
            'AgeAtDeath', 'Age_std', 'AgeUncertainty', 'SexEstimate', 'SexMale', 'Region'
        ]
        empty_df = pd.DataFrame({c: pd.Series(dtype="float") for c in keep_cols})
        # Ensure string columns are string dtype
        for s_col in ['SpecimenID', 'Genus', 'ToothClass', 'Region']:
            if s_col in empty_df.columns:
                empty_df[s_col] = empty_df[s_col].astype("string")
        return empty_df

    df = df.dropna(subset=required_cols)

    # Ensure SocketCount is numeric and positive integer; drop impossible rows
    df['SocketCount'] = pd.to_numeric(df['SocketCount'], errors='coerce')
    df = df[df['SocketCount'] > 0]

    # Ensure MissingCount is numeric and non-negative
    df['MissingCount'] = pd.to_numeric(df['MissingCount'], errors='coerce')
    df['MissingCount'] = df['MissingCount'].clip(lower=0)

    # Clip MissingCount so it does not exceed SocketCount and round to integer counts
    # Use a safe vectorized approach
    df['MissingCount'] = np.minimum(df['MissingCount'], df['SocketCount'])
    # Fill any remaining NaN MissingCount with 0 (should be none after dropna above)
    df['MissingCount'] = df['MissingCount'].fillna(0)
    # Round/truncate to integer counts
    df['MissingCount'] = df['MissingCount'].astype(int)
    df['SocketCount'] = df['SocketCount'].astype(int)

    # Recompute SexMale as integer 0/1
    df['SexMale'] = (df['SexEstimate'] >= 0.5).astype(int)

    # Standardize age for the model (handle constant or empty series)
    if not df['AgeAtDeath'].empty:
        age_mean = df['AgeAtDeath'].mean()
        age_std = df['AgeAtDeath'].std(ddof=0)
        if not np.isfinite(age_std) or age_std == 0:
            age_std = 1.0
        df['Age_std'] = (df['AgeAtDeath'] - age_mean) / age_std
    else:
        df['Age_std'] = np.nan

    # Create proportion column (useful for quick checks). Keep as float.
    # Avoid division by zero (SocketCount > 0 enforced above)
    df['PropMissing'] = df['MissingCount'] / df['SocketCount']

    # Ensure Region is trimmed (already cleaned above)
    df['Region'] = clean_text_series(df['Region'])

    # Keep only columns needed for modeling + SpecimenID for reference
    keep_cols = [
        'SpecimenID', 'MissingCount', 'SocketCount', 'PropMissing', 'Genus', 'ToothClass',
        'AgeAtDeath', 'Age_std', 'AgeUncertainty', 'SexEstimate', 'SexMale', 'Region'
    ]
    # Ensure we only select columns that exist (they should, but guard anyway)
    keep_existing = [c for c in keep_cols if c in df.columns]
    df = df[keep_existing]

    # Final safety: drop rows where SocketCount <= 0 or MissingCount < 0 (should be none)
    if 'SocketCount' in df.columns and 'MissingCount' in df.columns:
        df = df[(df['SocketCount'] > 0) & (df['MissingCount'] >= 0)]

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM (logit link) modeling the proportion of missing teeth (AMTL) as a function
    of Genus (categorical; primary IV), controlling for age, sex, tooth class, age uncertainty, and region.

    The model uses the observed SocketCount as frequency weights so counts are modeled as binomial trials.

    Returns the fitted statsmodels results object, or None if the input data are insufficient.
    """
    # Work on a copy
    df = df.copy()

    # Basic sanity checks
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if df.empty:
        warnings.warn("Input dataframe is empty. No data available to fit the model. Returning None.", UserWarning)
        return None

    # Ensure required columns exist
    required = ['MissingCount', 'SocketCount', 'Genus', 'ToothClass', 'Age_std', 'SexMale', 'AgeUncertainty', 'Region']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        warnings.warn(f"Input dataframe is missing required columns for modeling: {missing_cols}. Returning None.", UserWarning)
        return None

    # Drop rows with missing required data
    df = df.dropna(subset=required)
    if df.empty:
        warnings.warn("No complete cases remain after dropping rows with missing required variables. Returning None.", UserWarning)
        return None

    # Ensure numeric types for counts
    df['SocketCount'] = pd.to_numeric(df['SocketCount'], errors='coerce')
    df['MissingCount'] = pd.to_numeric(df['MissingCount'], errors='coerce')

    # Drop any rows that became NA after coercion
    df = df.dropna(subset=['SocketCount', 'MissingCount'])
    if df.empty:
        warnings.warn("No valid rows remain after coercing counts to numeric types. Returning None.", UserWarning)
        return None

    # Ensure integer counts and positive socket counts
    try:
        df['SocketCount'] = df['SocketCount'].astype(int)
        df['MissingCount'] = df['MissingCount'].astype(int)
    except Exception:
        # If casting to int fails for some reason, coerce safely
        df['SocketCount'] = df['SocketCount'].round().astype(int)
        df['MissingCount'] = df['MissingCount'].round().astype(int)

    df = df[(df['SocketCount'] > 0) & (df['MissingCount'] >= 0)]
    if df.empty:
        warnings.warn("No rows with valid count data (SocketCount>0 and MissingCount>=0) remain for modeling. Returning None.", UserWarning)
        return None

    # Recompute PropMissing to ensure consistency
    df['PropMissing'] = df['MissingCount'] / df['SocketCount']

    # Ensure categorical variables are strings and trimmed; this avoids category dtypes with zero levels
    for col in ['Genus', 'ToothClass', 'Region']:
        df[col] = df[col].where(df[col].notna(), pd.NA)
        notna_mask = df[col].notna()
        if notna_mask.any():
            df.loc[notna_mask, col] = df.loc[notna_mask, col].astype(str).str.strip()
        df[col] = df[col].replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA})

    # Check that categorical variables have at least one non-missing level
    for col in ['Genus', 'ToothClass', 'Region']:
        levels = df[col].dropna().unique()
        if len(levels) == 0:
            warnings.warn(f"Categorical column '{col}' has no non-missing levels; cannot fit model. Returning None.", UserWarning)
            return None

    # Build formula: proportion ~ categorical genus + standardized age + sex + tooth class + age uncertainty + region
    formula = 'PropMissing ~ C(Genus) + Age_std + SexMale + C(ToothClass) + AgeUncertainty + C(Region)'

    # Define the GLM with binomial family.
    glm_model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())

    # Fit using frequency weights = number of sockets per observation (trials)
    # Some statsmodels versions accept freq_weights, others accept weights.
    try:
        results = glm_model.fit(freq_weights=df['SocketCount'], disp=False)
    except TypeError:
        results = glm_model.fit(weights=df['SocketCount'], disp=False)

    return results