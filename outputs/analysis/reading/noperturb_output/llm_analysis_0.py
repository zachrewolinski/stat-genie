from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform cleaning and feature engineering for the Reader View analysis.

    Output dataframe contains the columns used in the model (see conceptual variables):
      - log_speed
      - reader_view
      - dyslexia_bin
      - age_scaled
      - num_words_scaled
      - Flesch_Kincaid_scaled
      - english_native_bin
      - retake_trial
      - device
      - page_id
      - uuid
    """
    df = df.copy()

    # Keep rows with essential variables
    df = df.dropna(subset=['speed', 'reader_view', 'uuid'])

    # Ensure numeric types for key variables
    # Remove non-positive speeds (cannot log-transform)
    df = df[df['speed'] > 0]

    # Standardize/clean dyslexia indicator. Prefer existing dyslexia_bin (0/1); if missing, derive from 'dyslexia' flag
    if 'dyslexia_bin' in df.columns:
        # Some datasets may encode dyslexia_bin as floats; coerce to 0/1
        df['dyslexia_bin'] = df['dyslexia_bin'].fillna(0).astype(float).apply(lambda x: 1 if x == 1 else 0).astype(int)
    else:
        # If only 'dyslexia' (0/1/2) is present, convert any nonzero to 1
        if 'dyslexia' in df.columns:
            df['dyslexia_bin'] = df['dyslexia'].fillna(0).astype(float).apply(lambda x: 1 if x > 0 else 0).astype(int)
        else:
            # If no dyslexia info, create column of zeros (will effectively test main effect only)
            df['dyslexia_bin'] = 0
            df['dyslexia_bin'] = df['dyslexia_bin'].astype(int)

    # Ensure reader_view is integer 0/1
    df['reader_view'] = df['reader_view'].fillna(0).astype(int)

    # Log-transform speed (dependent variable)
    df['log_speed'] = np.log(df['speed'].astype(float))

    # Binary for english native (map Y/N -> 1/0). Missing -> 0
    if 'english_native' in df.columns:
        df['english_native_bin'] = df['english_native'].map({'Y': 1, 'N': 0})
        df['english_native_bin'] = df['english_native_bin'].fillna(0).astype(int)
    else:
        df['english_native_bin'] = 0
        df['english_native_bin'] = df['english_native_bin'].astype(int)

    # Ensure retake_trial is binary 0/1
    if 'retake_trial' in df.columns:
        df['retake_trial'] = df['retake_trial'].fillna(0).astype(int)
    else:
        df['retake_trial'] = 0
        df['retake_trial'] = df['retake_trial'].astype(int)

    # Standardize continuous covariates: age, num_words, Flesch_Kincaid
    for col in ['age', 'num_words', 'Flesch_Kincaid']:
        if col in df.columns:
            # use population std (ddof=0) for stable scaling
            mean = df[col].mean()
            std = df[col].std(ddof=0)
            # avoid division by zero
            if std == 0 or np.isnan(std):
                df[col + '_scaled'] = 0.0
            else:
                df[col + '_scaled'] = (df[col] - mean) / std
        else:
            df[col + '_scaled'] = 0.0

    # Device and page_id as categorical (kept in the dataframe; will be used as fixed effects in the model)
    if 'device' in df.columns:
        df['device'] = df['device'].astype('category')
    else:
        df['device'] = 'unknown'
        df['device'] = df['device'].astype('category')

    if 'page_id' in df.columns:
        df['page_id'] = df['page_id'].astype('category')
    else:
        df['page_id'] = 'page_unknown'
        df['page_id'] = df['page_id'].astype('category')

    # Ensure uuid is string (grouping variable for mixed model)
    df['uuid'] = df['uuid'].astype(str)

    # Final column set required by the model
    final_cols = [
        'uuid',
        'log_speed',
        'reader_view',
        'dyslexia_bin',
        'age_scaled',
        'num_words_scaled',
        'Flesch_Kincaid_scaled',
        'english_native_bin',
        'retake_trial',
        'device',
        'page_id'
    ]

    # Keep only rows that still have non-missing values for the essential columns used in modeling.
    # Note: some columns like age_scaled etc. are filled with 0.0 if missing, so ensure DV/IV/moderator present.
    df = df.dropna(subset=['log_speed', 'reader_view', 'dyslexia_bin', 'uuid'])

    # Return dataframe with the final columns (reset index to avoid issues with statsmodels grouping)
    return df[final_cols].reset_index(drop=True)


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a mixed-effects regression to test whether Reader View (reader_view) improves reading speed
    and whether that effect differs for readers with dyslexia (dyslexia_bin as moderator).

    Model specification (fixed effects):
      log_speed ~ reader_view * dyslexia_bin + age_scaled + num_words_scaled + Flesch_Kincaid_scaled
                   + english_native_bin + retake_trial + C(device) + C(page_id)

    Random effects: random intercept for each participant (uuid) to account for repeated measures.

    Returns the fitted results object from statsmodels.
    """
    # Ensure required columns exist
    required = [
        'log_speed',
        'reader_view',
        'dyslexia_bin',
        'age_scaled',
        'num_words_scaled',
        'Flesch_Kincaid_scaled',
        'english_native_bin',
        'retake_trial',
        'device',
        'page_id',
        'uuid'
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column for modeling: {c}")

    # Make a working copy and drop rows with any missing values in the variables used by the model.
    model_cols = required.copy()
    df_model = df.dropna(subset=model_cols).copy()

    if df_model.shape[0] == 0:
        raise ValueError("No rows available for modeling after dropping missing values in required columns.")

    # Reset index to ensure statsmodels group indexing works correctly
    df_model = df_model.reset_index(drop=True)

    # Ensure appropriate dtypes for modeling
    # categorical fixed effects
    df_model['device'] = df_model['device'].astype('category')
    df_model['page_id'] = df_model['page_id'].astype('category')

    # Ensure binary/int columns are numeric ints
    for col in ['reader_view', 'dyslexia_bin', 'english_native_bin', 'retake_trial']:
        df_model[col] = pd.to_numeric(df_model[col]).astype(int)

    # Ensure uuid is string (grouping)
    df_model['uuid'] = df_model['uuid'].astype(str)

    # Build formula with interaction term and controls. C(device) and C(page_id) include categorical fixed effects.
    formula = (
        "log_speed ~ reader_view * dyslexia_bin + age_scaled + num_words_scaled + Flesch_Kincaid_scaled"
        " + english_native_bin + retake_trial + C(device) + C(page_id)"
    )

    # Fit mixed linear model with random intercept by uuid
    md = smf.mixedlm(formula, df_model, groups=df_model['uuid'], re_formula="~1")
    try:
        results = md.fit(reml=False, method='lbfgs')
    except Exception:
        # fallback to default fit if lbfgs fails
        results = md.fit(reml=False)

    # Print a concise summary for quick inspection
    print(results.summary())

    return results