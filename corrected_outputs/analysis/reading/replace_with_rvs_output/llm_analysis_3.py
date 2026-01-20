from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Example top-level read (kept for context; users can replace path or supply df directly)
# df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/reading/replace_with_rvs_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure critical columns exist
    required_cols = ['speed', 'reader_view', 'dyslexia_bin', 'uuid', 'page_id']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the dataframe")

    # Cast/clean critical columns
    # reader_view and dyslexia_bin should be numeric 0/1
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce')
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce')

    # Normalize reader_view and dyslexia_bin to explicit 0/1 where possible
    # If values are numeric, treat >=1 as 1, 0 as 0; leave NaN as-is
    df['reader_view'] = df['reader_view'].apply(lambda x: 1 if pd.notna(x) and x >= 1 else (0 if pd.notna(x) and x == 0 else np.nan))
    df['dyslexia_bin'] = df['dyslexia_bin'].apply(lambda x: 1 if pd.notna(x) and x >= 1 else (0 if pd.notna(x) and x == 0 else np.nan))

    # Keep uuid and page_id as strings for grouping/fixed-effects
    df['uuid'] = df['uuid'].astype(str)
    df['page_id'] = df['page_id'].astype(str)

    # Device and gender as categorical for formula-based modeling (if present)
    if 'device' in df.columns:
        df['device'] = df['device'].astype('category')
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype('category')

    # Map english_native to binary indicator (1 = Y, 0 = N); preserve missing as NaN
    if 'english_native' in df.columns:
        df['english_native_bin'] = df['english_native'].map({'Y': 1, 'N': 0})
    else:
        df['english_native_bin'] = np.nan

    # Ensure numeric speed and positive
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df = df.dropna(subset=['speed'])
    df = df[df['speed'] > 0].copy()

    # Log-transform the dependent variable to reduce skew
    df['log_speed'] = np.log(df['speed'])

    # Ensure numeric covariates are numeric; coerce errors to NaN then (optionally) drop if they exist
    numeric_covs = ['age', 'Flesch_Kincaid', 'num_words', 'retake_trial', 'correct_rate']
    for col in numeric_covs:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # If some of these covariates exist, drop rows with missing values on them to keep model estimation stable.
    covs_to_check = ['age', 'Flesch_Kincaid', 'num_words', 'retake_trial', 'correct_rate']
    existing_covs = [c for c in covs_to_check if c in df.columns]
    if existing_covs:
        df = df.dropna(subset=existing_covs)

    # Final sanity: if dyslexia column exists but dyslexia_bin does not (shouldn't happen due to earlier check),
    # create dyslexia_bin from dyslexia (map >=1 to 1)
    if 'dyslexia_bin' not in df.columns and 'dyslexia' in df.columns:
        df['dyslexia_bin'] = df['dyslexia'].apply(lambda x: 1 if pd.notna(x) and x >= 1 else 0)

    # Ensure all conceptual-final columns exist in the returned dataframe.
    # If any are missing, create them with appropriate types (filled with NaN/defaults) so downstream code can rely on their presence.
    conceptual_cols = [
        'reader_view', 'log_speed', 'dyslexia_bin', 'age', 'device', 'english_native_bin',
        'Flesch_Kincaid', 'num_words', 'retake_trial', 'correct_rate', 'gender',
        'page_id', 'uuid'
    ]
    for col in conceptual_cols:
        if col not in df.columns:
            # Create a column of NaNs or appropriate dtype placeholder
            if col in ('device', 'gender'):
                # categorical placeholder
                df[col] = pd.Series([pd.NA] * len(df), index=df.index, dtype='category')
            elif col in ('uuid', 'page_id'):
                # keep as string type placeholder but allow grouping code to detect missing later
                df[col] = pd.Series([pd.NA] * len(df), index=df.index).astype('object')
            else:
                df[col] = np.nan

    # Ensure categorical types for device and gender
    if 'device' in df.columns and df['device'].dtype != 'category':
        try:
            df['device'] = df['device'].astype('category')
        except Exception:
            df['device'] = df['device'].astype('category')

    if 'gender' in df.columns and df['gender'].dtype != 'category':
        try:
            df['gender'] = df['gender'].astype('category')
        except Exception:
            df['gender'] = df['gender'].astype('category')

    # Ensure reader_view and dyslexia_bin are numeric (0/1) if possible
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce')
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce')

    # Reset index to ensure contiguous 0..n-1 indexing (important for mixedlm internals)
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Mixed effects regression: random intercepts for participant (uuid), page_id as fixed effect
    # We test the interaction reader_view * dyslexia_bin to see whether Reader View benefits readers with dyslexia.

    # Work on a copy and ensure contiguous indexing
    df = df.copy().reset_index(drop=True)

    # Check for critical columns
    critical = ['log_speed', 'reader_view', 'dyslexia_bin', 'uuid']
    missing_critical = [c for c in critical if c not in df.columns]
    if missing_critical:
        raise ValueError(f"Critical column(s) missing from dataframe: {missing_critical}")

    # Optionally ensure types are suitable
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce')
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce')
    df['log_speed'] = pd.to_numeric(df['log_speed'], errors='coerce')

    # Define candidate covariates and categoricals
    numeric_covs = ['age', 'Flesch_Kincaid', 'num_words', 'retake_trial', 'correct_rate', 'english_native_bin']
    categorical_covs = ['device', 'gender', 'page_id']

    # Build list of numeric covariates to potentially include (have >1 unique non-missing values)
    included_covariates = []
    for cov in numeric_covs:
        if cov in df.columns and df[cov].dropna().nunique() > 1:
            included_covariates.append(cov)

    # Helper to decide which categorical covariates to include given a particular dataframe
    def categorical_terms_for(df_sub):
        cats = []
        for cat in categorical_covs:
            if cat in df_sub.columns and df_sub[cat].dropna().nunique() > 1:
                cats.append(f'C({cat})')
        return cats

    # Attempt 1: Fit full model with interaction if data permit
    full_vars_needed = ['log_speed', 'reader_view', 'dyslexia_bin', 'uuid'] + included_covariates
    df_full = df.dropna(subset=full_vars_needed).reset_index(drop=True)

    def fit_mixedlm(formula, df_fit):
        # Ensure categorical types for any C(...) variables in the dataframe
        for cat in categorical_covs:
            if cat in df_fit.columns and df_fit[cat].dropna().nunique() > 1:
                df_fit[cat] = df_fit[cat].astype('category')
        md = smf.mixedlm(formula, df_fit, groups=df_fit['uuid'])
        return md.fit(reml=False)

    # Try full interaction model if both reader_view and dyslexia_bin vary in df_full
    if df_full.shape[0] > 0 and df_full['reader_view'].nunique(dropna=True) > 1 and df_full['dyslexia_bin'].nunique(dropna=True) > 1:
        # compose formula with numeric covariates and categorical terms based on df_full
        formula_terms = ['reader_view * dyslexia_bin'] + included_covariates + categorical_terms_for(df_full)
        formula = 'log_speed ~ ' + ' + '.join(formula_terms)
        try:
            mdf = fit_mixedlm(formula, df_full)
            return mdf
        except Exception:
            # fall through to try reduced models
            pass

    # Attempt 2: If dyslexia_bin has no variation (constant or missing), fit reduced model without dyslexia_bin (no interaction)
    reduced_included_covs = included_covariates[:]  # numeric covs to include
    # We will not include dyslexia_bin in reduced model
    reduced_vars_needed = ['log_speed', 'reader_view', 'uuid'] + reduced_included_covs
    df_reduced = df.dropna(subset=reduced_vars_needed).reset_index(drop=True)

    if df_reduced.shape[0] > 0 and df_reduced['reader_view'].nunique(dropna=True) > 1:
        # include categorical terms based on df_reduced
        cat_terms = categorical_terms_for(df_reduced)
        formula_terms = ['reader_view'] + reduced_included_covs + cat_terms
        formula = 'log_speed ~ ' + ' + '.join(formula_terms)
        try:
            mdf = fit_mixedlm(formula, df_reduced)
            return mdf
        except Exception:
            # fall through to simplest fallback
            pass

    # Final fallback: simplest model with only reader_view (and uuid groups)
    fallback_vars = ['log_speed', 'reader_view', 'uuid']
    df_fallback = df.dropna(subset=fallback_vars).reset_index(drop=True)
    if df_fallback.shape[0] == 0:
        raise ValueError("No data available for any model after dropping missing values required for the model.")
    if df_fallback['reader_view'].nunique(dropna=True) < 2:
        raise ValueError("reader_view has no variation (constant) after filtering; cannot fit the model.")

    fallback_formula = 'log_speed ~ reader_view'
    md2 = smf.mixedlm(fallback_formula, df_fallback, groups=df_fallback['uuid'])
    mdf = md2.fit(reml=False)
    return mdf