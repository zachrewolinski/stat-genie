from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Optional: example top-level read (kept from original context). Users may replace path as needed.
# Commented out to avoid file-not-found errors when importing this module in other environments.
# df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/reading/noperturb_output/reading.csv')


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe. The function:
      - drops rows missing key variables (speed, reader_view, dyslexia_bin)
      - coerces binary/categorical variables into usable forms
      - creates a log-transformed, winsorized dependent variable 'log_speed'
      - standardizes continuous covariates (age, Flesch_Kincaid, num_words) into *_z columns
      - fills or coerces other covariates

    Returns dataframe containing all columns listed in the conceptual variables.
    """
    df = df.copy()

    # Ensure core columns exist; if they do not exist, raise an informative error.
    core_required = ['speed', 'reader_view', 'dyslexia_bin']
    missing_core = [c for c in core_required if c not in df.columns]
    if missing_core:
        raise ValueError(f"Input dataframe missing required core columns: {missing_core}")

    # Drop rows missing core variables required for the model
    df = df.dropna(subset=['speed', 'reader_view', 'dyslexia_bin'])

    # Ensure binary indicators are integers (if convertible)
    # If values are non-numeric (e.g., 'Y'/'N'), attempt mapping first is not attempted here since these
    # fields are expected numeric/binary already per contract.
    df['reader_view'] = df['reader_view'].astype(int)
    df['dyslexia_bin'] = df['dyslexia_bin'].astype(int)

    # English native -> binary (Y -> 1, N -> 0). If english_native column missing, create english_native_bin = 0.
    if 'english_native' in df.columns:
        df['english_native_bin'] = df['english_native'].map({'Y': 1, 'N': 0})
        df['english_native_bin'] = df['english_native_bin'].fillna(0).astype(int)
    else:
        df['english_native_bin'] = 0

    # Retake and correct_rate: fill missing reasonable defaults or create if absent
    if 'retake_trial' in df.columns:
        df['retake_trial'] = df['retake_trial'].fillna(0).astype(int)
    else:
        df['retake_trial'] = 0

    if 'correct_rate' in df.columns:
        # If entirely missing values, fill with 0.0 (no correct answers)
        if df['correct_rate'].isna().all():
            df['correct_rate'] = 0.0
        else:
            df['correct_rate'] = df['correct_rate'].fillna(df['correct_rate'].median())
    else:
        df['correct_rate'] = 0.0

    # Device and page_id: fill missing with explicit category
    if 'device' in df.columns:
        df['device'] = df['device'].fillna('unknown').astype(str)
    else:
        df['device'] = 'unknown'

    if 'page_id' in df.columns:
        df['page_id'] = df['page_id'].fillna('unknown').astype(str)
    else:
        df['page_id'] = 'unknown'

    # Ensure uuid exists; if not, create a unique id per row so clustering is still possible.
    if 'uuid' not in df.columns:
        # create integer ids for each row
        df['uuid'] = np.arange(len(df)).astype(int)
    else:
        # If uuid has missing values, fill with unique row-based ids for those rows
        if df['uuid'].isna().any():
            na_mask = df['uuid'].isna()
            # create new ids that won't collide with existing ones by using a high offset
            try:
                existing_max = pd.to_numeric(df.loc[~na_mask, 'uuid'], errors='coerce').dropna().astype(float).max()
                if np.isfinite(existing_max):
                    offset = int(existing_max) + 1
                else:
                    offset = len(df)
            except Exception:
                offset = len(df)
            df.loc[na_mask, 'uuid'] = np.arange(offset, offset + na_mask.sum())
        # keep uuid as-is (could be string or int); do not coerce here since model() will encode groups properly.

    # Create log-transformed speed to stabilize variance and reduce effect of outliers
    # speed > 0 in this dataset; add small epsilon and take log
    df['log_speed'] = np.log(df['speed'].clip(lower=1e-3))

    # Winsorize the log_speed to the 1st and 99th percentiles to limit extreme influence
    lower, upper = df['log_speed'].quantile([0.01, 0.99])
    df['log_speed'] = df['log_speed'].clip(lower=lower, upper=upper)

    # Standardize continuous controls (z-scores). Use population std (ddof=0) for stability.
    for col in ['age', 'Flesch_Kincaid', 'num_words']:
        zcol = col + '_z'
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std(ddof=0)
            # If std is zero (constant) or NaN, create zero column
            if std == 0 or np.isnan(std):
                df[zcol] = 0.0
            else:
                df[zcol] = (df[col] - mean) / std
        else:
            # If missing entirely, create a zero column
            df[zcol] = 0.0

    # Keep only columns necessary for modeling (but also preserve uuid and page_id)
    # Columns enumerated in conceptual variables:
    required_cols = [
        'uuid', 'page_id', 'reader_view', 'dyslexia_bin', 'log_speed',
        'age_z', 'device', 'english_native_bin', 'Flesch_Kincaid_z',
        'num_words_z', 'retake_trial', 'correct_rate'
    ]

    # If any required column is missing in df for unexpected reason, create placeholder
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Final drop: ensure DV has no missing values and the key IV/moderator exist
    df = df.dropna(subset=['log_speed', 'reader_view', 'dyslexia_bin'])

    # Ensure types of required boolean/binary columns are ints (where reasonable)
    try:
        df['reader_view'] = df['reader_view'].astype(int)
    except Exception:
        # leave as-is if cannot convert; model() will handle encoding
        pass
    try:
        df['dyslexia_bin'] = df['dyslexia_bin'].astype(int)
    except Exception:
        pass
    try:
        df['retake_trial'] = df['retake_trial'].astype(int)
    except Exception:
        pass
    try:
        df['english_native_bin'] = df['english_native_bin'].astype(int)
    except Exception:
        pass

    # Ensure page_id and device are strings
    df['page_id'] = df['page_id'].astype(str)
    df['device'] = df['device'].astype(str)

    # Ensure final dataframe contains exactly the conceptual required columns plus any extras preserved
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model on the transformed dataframe testing whether Reader View improves reading speed
    differently for readers with vs. without dyslexia. The model uses clustered robust standard errors
    at the participant level (uuid) to account for repeated measures.

    Model formula:
      log_speed ~ reader_view * dyslexia_bin
                  + age_z + Flesch_Kincaid_z + num_words_z
                  + retake_trial + correct_rate + english_native_bin
                  + C(device) + C(page_id)

    Returns the fitted statsmodels results object (OLSResults) with clustered SEs.
    """
    # Ensure the required columns are present
    cols_needed = [
        'log_speed', 'reader_view', 'dyslexia_bin', 'age_z', 'Flesch_Kincaid_z',
        'num_words_z', 'retake_trial', 'correct_rate', 'english_native_bin', 'device',
        'page_id', 'uuid'
    ]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Define formula with interaction between reader_view and dyslexia_bin
    formula = (
        'log_speed ~ reader_view * dyslexia_bin + age_z + Flesch_Kincaid_z + num_words_z '
        '+ retake_trial + correct_rate + english_native_bin + C(device) + C(page_id)'
    )

    # Fit OLS first to allow Patsy to drop rows with missing data and to obtain the row mapping
    ols_mod = smf.ols(formula, data=df)
    ols_results = ols_mod.fit()

    # Obtain the original-row labels that Patsy kept for the model (these correspond to df indices)
    try:
        row_labels = ols_results.model.data.row_labels
    except Exception:
        # Fallback: assume all rows were used
        row_labels = df.index.values

    # Align cluster groups to the rows actually used by the model
    groups_aligned = pd.Categorical(df.loc[row_labels, 'uuid']).codes

    # Now obtain clustered robust covariance results based on the aligned groups
    results = ols_results.get_robustcov_results(cov_type='cluster', groups=groups_aligned)

    # Return fitted results object (callers can use .summary(), .params, .pvalues, etc.)
    return results