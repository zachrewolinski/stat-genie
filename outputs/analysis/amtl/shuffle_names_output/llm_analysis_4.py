from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and rename input columns to consistent, analysis-ready columns.

    The function maps messy/mislabelled input columns to the required final
    column names used by the model:
      - 'Genus' (taxonomic genus)
      - 'Specimen' (identifier)
      - 'Missing' (count of missing teeth = AMTL)
      - 'Sockets' (number of observable sockets)
      - 'Age' (estimated age at death)
      - 'Age_sd' (assigned uncertainty of age at death)
      - 'ProbMale' (estimated probability specimen is male)
      - 'ToothClass' (tooth class: Anterior/Posterior/Premolar)
      - plus derived diagnostics like 'AMTL_rate' and optional centered helpers

    The function is defensive: it only renames columns that exist, coerces
    types, ensures counts are integers within valid ranges, and returns a
    dataframe containing the required final columns (and additional helpers).
    """
    df = df.copy()

    # Map common raw column name variants (lowercased) to required final column names.
    # When multiple raw columns map to the same final name, the first encountered is used.
    lower_to_final = {
        # Genus
        'genus': 'Genus',
        # Specimen / identifier
        'specimen': 'Specimen',
        'id': 'Specimen',
        'sample_id': 'Specimen',
        # Missing teeth / AMTL
        'missing': 'Missing',
        'amtl': 'Missing',
        'num_amtl': 'Missing',
        'num_missing': 'Missing',
        'n_missing': 'Missing',
        # Sockets / trials
        'sockets': 'Sockets',
        'n_sockets': 'Sockets',
        'num_sockets': 'Sockets',
        # Age
        'age': 'Age',
        'estimated_age': 'Age',
        'est_age': 'Age',
        # Age sd / uncertainty
        'age_sd': 'Age_sd',
        'stdev_age': 'Age_sd',
        'sd_age': 'Age_sd',
        'age_se': 'Age_sd',
        # ProbMale
        'prob_male': 'ProbMale',
        'probmale': 'ProbMale',
        'male_prob': 'ProbMale',
        'p_male': 'ProbMale',
        # Tooth class
        'tooth_class': 'ToothClass',
        'toothclass': 'ToothClass',
        'tooth': 'ToothClass',
        'class': 'ToothClass',
        # Region/population (auxiliary)
        'region': 'Region',
        'pop': 'Region',
        'population': 'Region',
    }

    # Build rename mapping based on actual columns present (preserve original casing of source columns)
    existing_rename = {}
    used_final_names = set()
    for col in df.columns:
        key = str(col).strip().lower()
        if key in lower_to_final:
            final_name = lower_to_final[key]
            # Avoid overwriting if multiple raw cols map to same final name; keep first encounter.
            if final_name not in used_final_names:
                existing_rename[col] = final_name
                used_final_names.add(final_name)

    if existing_rename:
        df = df.rename(columns=existing_rename)

    # Ensure all required final columns exist; create with NaN if missing so downstream code can handle.
    required_final = ['Specimen', 'Genus', 'Missing', 'Sockets', 'Age', 'Age_sd', 'ProbMale', 'ToothClass']
    for col in required_final:
        if col not in df.columns:
            df[col] = np.nan

    # Normalize string columns (only operate on non-missing values to preserve NaN)
    for col in ['Genus', 'ToothClass', 'Specimen', 'Region']:
        if col in df.columns:
            df[col] = df[col].astype(object)
            non_na = df[col].notna()
            if non_na.any():
                df.loc[non_na, col] = df.loc[non_na, col].astype(str).str.strip()

    # Numeric conversions for counts and continuous controls
    for col in ['Missing', 'Sockets', 'Age', 'Age_sd', 'ProbMale']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows missing critical identifiers or socket info early (we require these)
    df = df.dropna(subset=['Specimen', 'Genus', 'ToothClass', 'Sockets'], how='any')

    # Round/count handling: Sockets and Missing are counts
    # Round Sockets and convert to integer; negative sockets are invalid -> set to 0 and will be dropped
    df['Sockets'] = df['Sockets'].round()
    # Any remaining NaN in Sockets would have been dropped above; coerce to int safely
    df['Sockets'] = df['Sockets'].astype(int)

    # Require at least 1 socket to be an observation
    df = df[df['Sockets'] > 0].copy()

    # Missing rounding and bounds
    df['Missing'] = df['Missing'].round().fillna(0).astype(int)
    df.loc[df['Missing'] < 0, 'Missing'] = 0
    # Cap Missing at Sockets
    over_mask = df['Missing'] > df['Sockets']
    if over_mask.any():
        df.loc[over_mask, 'Missing'] = df.loc[over_mask, 'Sockets']

    # ProbMale normalization: if reported as 0-100, convert to 0-1
    if 'ProbMale' in df.columns and df['ProbMale'].notna().any():
        med = df['ProbMale'].median(skipna=True)
        if pd.notna(med) and med > 1.0:
            df['ProbMale'] = df['ProbMale'] / 100.0
    if 'ProbMale' in df.columns:
        df['ProbMale'] = df['ProbMale'].clip(lower=0.0, upper=1.0)

    # AMTL rate diagnostic
    df['AMTL_rate'] = df['Missing'] / df['Sockets']

    # Create centered helper columns for convenience (but the model will use canonical names)
    # Use mean with skipna=True so NaNs do not propagate to all rows
    if 'Age' in df.columns:
        age_mean = df['Age'].mean(skipna=True)
        df['Age_c'] = df['Age'] - age_mean
    else:
        df['Age_c'] = np.nan

    if 'ProbMale' in df.columns:
        pm_mean = df['ProbMale'].mean(skipna=True)
        df['ProbMale_c'] = df['ProbMale'] - pm_mean
    else:
        df['ProbMale_c'] = np.nan

    if 'Age_sd' in df.columns:
        asd_mean = df['Age_sd'].mean(skipna=True)
        df['Age_sd_c'] = df['Age_sd'] - asd_mean
    else:
        df['Age_sd_c'] = np.nan

    # Ensure required conceptual variables are present (drop rows missing them)
    df = df.dropna(subset=['Missing', 'Sockets', 'Age', 'ProbMale', 'Age_sd', 'Specimen', 'Genus', 'ToothClass'], how='any')

    # Final columns to return: keep required conceptual columns plus useful diagnostics/helpers
    keep_cols = ['Specimen', 'Genus', 'Missing', 'Sockets', 'Age', 'ProbMale', 'ToothClass', 'Age_sd',
                 'AMTL_rate', 'Age_c', 'ProbMale_c', 'Age_sd_c']
    present_keep = [c for c in keep_cols if c in df.columns]
    return df[present_keep].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a binomial GLM (logit link) predicting AMTL (Missing out of Sockets)
    from Genus while controlling for Age, ProbMale, ToothClass, and Age_sd.
    Standard errors are clustered by Specimen.

    The function expects the FINAL dataframe produced by transform() with the
    exact required columns:
      ['Specimen', 'Genus', 'Missing', 'Sockets', 'Age', 'ProbMale', 'ToothClass', 'Age_sd']

    Returns the fitted (possibly clustered robust) results object.
    """
    data = df.copy()

    # Verify presence of required columns
    required = ['Specimen', 'Genus', 'Missing', 'Sockets', 'Age', 'ProbMale', 'ToothClass', 'Age_sd']
    missing_cols = [c for c in required if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Input dataframe is missing required columns: {missing_cols}")

    # Drop any rows with missing values in required model inputs (safety)
    data = data.dropna(subset=required, how='any')
    if data.shape[0] == 0:
        raise ValueError("No valid observations available for modeling after dropping missing data.")

    # Prepare categorical predictors
    data['Genus'] = data['Genus'].astype('category')
    data['ToothClass'] = data['ToothClass'].astype('category')

    # Categorical dummies (drop_first to avoid collinearity)
    X_genus = pd.get_dummies(data['Genus'], prefix='Genus', drop_first=True)
    X_tooth = pd.get_dummies(data['ToothClass'], prefix='ToothClass', drop_first=True)
    X_cats = pd.concat([X_genus, X_tooth], axis=1)

    # Continuous covariates: use the canonical column names. Center them for stability (internal operation).
    X_cont = data[['Age', 'ProbMale', 'Age_sd']].astype(float).copy()
    X_cont = X_cont - X_cont.mean()

    # Combine predictors
    X = pd.concat([X_cont, X_cats], axis=1)

    # Ensure there is at least a constant column for the model
    if X.shape[1] == 0:
        X = pd.DataFrame({'const': np.ones(len(data))}, index=data.index)
    else:
        X = sm.add_constant(X, has_constant='add')

    # Response and weights
    # Use proportion as endog and sockets as frequency weights
    y = (data['Missing'] / data['Sockets']).astype(float)
    weights = data['Sockets'].astype(float)

    # Align observations: drop any rows with NA in y, weights, or X
    model_df = pd.concat([y.rename('y'), weights.rename('w'), X], axis=1)
    model_df = model_df.dropna()
    if model_df.shape[0] == 0:
        raise ValueError("No valid observations available for modeling after dropping missing data.")

    # Reconstruct y, weights, X aligned
    y_aligned = model_df['y']
    weights_aligned = model_df['w']
    X_aligned = model_df.drop(columns=['y', 'w'])

    # If after alignment X_aligned has zero columns, add intercept
    if X_aligned.shape[1] == 0:
        X_aligned = pd.DataFrame({'const': np.ones(len(model_df))}, index=model_df.index)

    # Fit GLM with Binomial family using frequency weights (Sockets)
    glm_model = sm.GLM(y_aligned, X_aligned, family=sm.families.Binomial(), freq_weights=weights_aligned)
    res = glm_model.fit()

    # Compute clustered robust covariance if possible (cluster by Specimen aligned to model_df index)
    if 'Specimen' in data.columns:
        clusters = data.loc[model_df.index, 'Specimen']
        try:
            clustered_res = res.get_robustcov_results(cov_type='cluster', groups=clusters)
        except Exception:
            clustered_res = res
    else:
        clustered_res = res

    print(clustered_res.summary())
    return clustered_res