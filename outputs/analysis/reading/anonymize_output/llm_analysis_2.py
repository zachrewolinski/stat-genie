from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/reading/anonymize_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Standardize column access / create core variables
    # Reader view indicator (feature3): 1 == activated, 0 == not
    df['ReaderView'] = pd.to_numeric(df['feature3'], errors='coerce').astype(float)

    # Primary reading time (prefer feature5 which is time minus scrolling). Fall back to feature4 - feature6
    df['ReadingTimeMS'] = pd.to_numeric(df['feature5'], errors='coerce')
    # where missing, compute as total - scroll
    mask_missing_rt = df['ReadingTimeMS'].isna() & df['feature4'].notna()
    df.loc[mask_missing_rt, 'ReadingTimeMS'] = (
        pd.to_numeric(df.loc[mask_missing_rt, 'feature4'], errors='coerce')
        - pd.to_numeric(df.loc[mask_missing_rt, 'feature6'], errors='coerce')
    )

    # Words on the page
    df['Words'] = pd.to_numeric(df['feature7'], errors='coerce')

    # Basic engagement measure: comprehension accuracy
    df['Comprehension'] = pd.to_numeric(df['feature8'], errors='coerce')

    # Dyslexia status: use feature17 as primary binary indicator (0/1). If missing, try to derive from feature12.
    df['Dyslexia'] = pd.to_numeric(df['feature17'], errors='coerce')
    # feature17 is 1 if dyslexia, 0 if not; if feature17 missing but feature12 present, treat feature12>0 as dyslexia
    fallback_mask = df['Dyslexia'].isna() & df['feature12'].notna()
    df.loc[fallback_mask, 'Dyslexia'] = (
        pd.to_numeric(df.loc[fallback_mask, 'feature12'], errors='coerce') > 0
    ).astype(float)

    # Dyslexia severity (0/1/2) from feature12 (keep as numeric if present)
    df['DyslexiaSeverity'] = pd.to_numeric(df['feature12'], errors='coerce')

    # Other controls
    # Use object dtype (plain Python strings) rather than pandas' "string" dtype so patsy/statsmodels can handle them
    df['Device'] = df['feature11'].astype(object)
    df['Age'] = pd.to_numeric(df['feature10'], errors='coerce')
    df['Education'] = df['feature13'].astype(object)
    df['Gender'] = pd.to_numeric(df['feature14'], errors='coerce')
    # Native English: feature18 'Y'/'N'
    df['NativeEnglish'] = df['feature18'].map({'Y': 1, 'N': 0})
    df['NativeEnglish'] = pd.to_numeric(df['NativeEnglish'], errors='coerce').astype(float)
    df['FleschKincaid'] = pd.to_numeric(df['feature19'], errors='coerce')
    df['Retake'] = pd.to_numeric(df['feature16'], errors='coerce')

    # Page and record ids (object dtype)
    df['PageID'] = df['feature2'].astype(object)
    df['RecordID'] = df['feature1'].astype(object)

    # Compute reading speed: words / reading time (in seconds)
    df['ReadingTimeS'] = df['ReadingTimeMS'] / 1000.0
    # Remove unrealistic values: require positive words and at least 0.2s reading time
    df = df[df['Words'].notna() & df['ReadingTimeS'].notna()]
    df = df[(df['Words'] > 0) & (df['ReadingTimeS'] >= 0.2)]

    df['ReadingSpeed_WPS'] = df['Words'] / df['ReadingTimeS']

    # Drop nonpositive or infinite speeds
    # Ensure numeric float type for isfinite checks
    df = df[df['ReadingSpeed_WPS'].notna()]
    df = df[np.isfinite(df['ReadingSpeed_WPS'].astype(float)) & (df['ReadingSpeed_WPS'] > 0)]

    # Remove extreme outliers in reading speed: > mean + 4*sd
    rs_mean = df['ReadingSpeed_WPS'].mean()
    rs_std = df['ReadingSpeed_WPS'].std()
    if pd.notna(rs_mean) and pd.notna(rs_std):
        df = df[df['ReadingSpeed_WPS'] <= (rs_mean + 4 * rs_std)]

    # Log-transform the reading speed to stabilize skew
    # Guard against nonpositive values just in case
    df = df[df['ReadingSpeed_WPS'] > 0]
    df['LogReadingSpeed'] = np.log(df['ReadingSpeed_WPS'].astype(float))

    # Ensure ReaderView and Dyslexia are integer/binary 0/1 where possible
    df['ReaderView'] = df['ReaderView'].fillna(0).astype(int)
    df['Dyslexia'] = df['Dyslexia'].fillna(0).astype(int)

    # Recompute a small diagnostics column (optional) - proportion missing important controls
    df['__missing_controls'] = (df[['Age', 'Device', 'Education', 'FleschKincaid']].isna().sum(axis=1))

    # Enforce numpy-backed dtypes for columns used by patsy/statsmodels (avoid pandas nullable dtypes)
    numeric_cols = [
        'ReadingTimeMS', 'ReadingTimeS', 'ReadingSpeed_WPS', 'LogReadingSpeed',
        'Words', 'Comprehension', 'FleschKincaid', 'Age', 'Gender', 'NativeEnglish', 'Retake'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

    # Ensure categorical/object columns are plain Python objects
    object_cols = ['Device', 'Education', 'PageID', 'RecordID']
    for col in object_cols:
        if col in df.columns:
            df[col] = df[col].astype(object)

    # Keep only columns needed for analysis (plus some trace columns)
    keep_cols = [
        'RecordID', 'PageID', 'ReaderView', 'Dyslexia', 'DyslexiaSeverity',
        'Words', 'ReadingTimeMS', 'ReadingTimeS', 'ReadingSpeed_WPS', 'LogReadingSpeed',
        'Comprehension', 'FleschKincaid', 'Device', 'Age', 'Education', 'Gender', 'NativeEnglish', 'Retake',
        '__missing_controls'
    ]
    # Some columns may not exist if original data missing; intersect to avoid KeyError
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Make a local copy
    df = df.copy()

    # Basic checks
    required = ['LogReadingSpeed', 'ReaderView', 'Dyslexia', 'PageID']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column {c} not found in dataframe")

    # Formula: main effect of ReaderView, Dyslexia, and their interaction.
    # Control for age, gender, retake, comprehension, flesch score, device, education, native speaker, and page fixed effects.
    formula = (
        'LogReadingSpeed ~ ReaderView * Dyslexia '
        '+ Age + Gender + Retake + Comprehension + FleschKincaid '
        '+ C(Device) + C(Education) + C(NativeEnglish) + C(PageID)'
    )

    # Prepare the dataframe that will actually be used in the model (drop rows with missing values
    # in any variable that appears in the formula). This ensures the clustering groups array
    # we pass to statsmodels has the same length as the data used to estimate the model.
    model_vars = [
        'LogReadingSpeed', 'ReaderView', 'Dyslexia', 'Age', 'Gender', 'Retake',
        'Comprehension', 'FleschKincaid', 'Device', 'Education', 'NativeEnglish', 'PageID'
    ]
    model_df = df.copy()
    # Drop rows with NA in any of the variables used by the model
    model_df = model_df.dropna(subset=model_vars).reset_index(drop=True)

    # If PageID has missing values that weren't caught, drop them (shouldn't be any after dropna above).
    if model_df['PageID'].isna().any():
        model_df = model_df[model_df['PageID'].notna()].reset_index(drop=True)

    # Create integer cluster codes aligned with model_df rows.
    # pd.Categorical.codes produces -1 for NaN, but we've dropped NaNs already.
    groups = pd.Categorical(model_df['PageID']).codes
    if (groups < 0).any():
        # Drop any remaining problematic rows (defensive)
        mask_valid = groups >= 0
        model_df = model_df.loc[mask_valid].reset_index(drop=True)
        groups = pd.Categorical(model_df['PageID']).codes

    # Fit OLS with cluster-robust standard errors clustered by PageID (using integer codes aligned to model_df)
    results = smf.ols(formula, data=model_df).fit(cov_type='cluster', cov_kwds={'groups': groups})

    # Return the fitted results object (caller can print summary())
    return results