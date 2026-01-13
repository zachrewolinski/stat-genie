from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataframe for modeling.

    Transformations performed:
    - Work on a copy
    - Remove retake trials (retain only retake_trial == 0) because retakes may reflect non-standard behavior
    - Ensure required columns exist and drop rows with missing values in model variables
    - Ensure dyslexia_bin exists (if not, create from 'dyslexia' > 0)
    - Keep only positive speed values and create log_speed = np.log(speed)
    - Cast categorical variables to category and map english_native to binary (1/0)
    - Ensure reader_view and dyslexia_bin are integer/binary
    - Return transformed dataframe containing at least the columns listed in the conceptual model
    """
    df = df.copy()

    # Drop obvious invalid / non-standard trials: keep only non-retake trials
    if 'retake_trial' in df.columns:
        df = df[df['retake_trial'] == 0]

    # Ensure dyslexia_bin exists; if not, derive from 'dyslexia' (values > 0 -> dyslexia)
    if 'dyslexia_bin' not in df.columns and 'dyslexia' in df.columns:
        df['dyslexia_bin'] = (df['dyslexia'] > 0).astype(int)

    # Required columns for model (these must be present in the final df)
    required = [
        'speed',
        'reader_view',
        'dyslexia_bin',
        'age',
        'device',
        'language',
        'english_native',
        'num_words',
        'Flesch_Kincaid',
        'correct_rate',
        'page_id',
        'uuid'
    ]

    # If any required column is completely missing from the DataFrame, we cannot proceed
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        # Return an empty dataframe with required columns to preserve contract (caller can detect emptiness)
        empty_df = pd.DataFrame(columns=required + ['log_speed'])
        return empty_df

    # Work on rows that have non-missing values for all required columns
    df = df.dropna(subset=required)

    # Ensure sensible numeric types and binary encodings

    # reader_view: try to coerce to binary 0/1
    if 'reader_view' in df.columns:
        # Handle common string encodings
        if df['reader_view'].dtype == object or df['reader_view'].dtype.name == 'category':
            df['reader_view'] = df['reader_view'].replace({'Y': 1, 'N': 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0})
            df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce')
        else:
            df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce')

        # Drop any rows where reader_view could not be interpreted as numeric
        df = df.dropna(subset=['reader_view'])
        # Cast to int (0/1 expected)
        df['reader_view'] = df['reader_view'].astype(int)

    # dyslexia_bin ensure int and binary
    if 'dyslexia_bin' in df.columns:
        if df['dyslexia_bin'].dtype == object or df['dyslexia_bin'].dtype.name == 'category':
            df['dyslexia_bin'] = df['dyslexia_bin'].replace({'Y': 1, 'N': 0, 'yes': 1, 'no': 0})
        df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce')
        df = df.dropna(subset=['dyslexia_bin'])
        df['dyslexia_bin'] = df['dyslexia_bin'].astype(int)

    # Remove nonpositive speeds and compute log transform
    if 'speed' in df.columns:
        df = df[pd.to_numeric(df['speed'], errors='coerce').notnull()]
        df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
        df = df[df['speed'] > 0]
        # log transform to stabilize variance / reduce skew
        df['log_speed'] = np.log(df['speed'])

    # Map english_native to 1/0 if it's 'Y'/'N' or similar
    if 'english_native' in df.columns:
        if df['english_native'].dtype == object or df['english_native'].dtype.name == 'category':
            df['english_native'] = df['english_native'].replace({'Y': 1, 'N': 0, 'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, True: 1, False: 0})
            df['english_native'] = pd.to_numeric(df['english_native'], errors='coerce')
        else:
            df['english_native'] = pd.to_numeric(df['english_native'], errors='coerce')

    # Cast category-like variables to category dtype for modeling convenience
    for cat_col in ['device', 'language', 'page_id']:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype('category')

    # Final dropna in case mapping introduced NaNs
    final_cols = [
        'log_speed',
        'reader_view',
        'dyslexia_bin',
        'age',
        'device',
        'language',
        'english_native',
        'num_words',
        'Flesch_Kincaid',
        'correct_rate',
        'page_id',
        'uuid'
    ]
    df = df.dropna(subset=final_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model to estimate whether Reader View improves reading speed for individuals with dyslexia.

    Model specification (primary):
      log_speed ~ reader_view * dyslexia_bin + age + C(device) + C(language) + english_native + num_words + Flesch_Kincaid + correct_rate + C(page_id)

    The interaction term reader_view:dyslexia_bin tests whether the Reader View effect differs for readers with dyslexia.
    Standard errors are clustered by participant (uuid) to account for within-subject correlation.

    Returns the fitted statsmodels result object, or None if the model cannot be estimated (e.g., no rows).
    """
    # Defensive checks: ensure df has required columns
    required_final = [
        'log_speed',
        'reader_view',
        'dyslexia_bin',
        'age',
        'device',
        'language',
        'english_native',
        'num_words',
        'Flesch_Kincaid',
        'correct_rate',
        'page_id',
        'uuid'
    ]
    missing = [c for c in required_final if c not in df.columns]
    if missing:
        # Cannot fit model if required columns are missing; return None to indicate inability to fit
        return None

    # If no rows, return None rather than attempting to fit (avoids patsy negative-dimension errors)
    if df.shape[0] == 0:
        return None

    # Ensure categorical columns are category dtype
    for cat_col in ['device', 'language', 'page_id']:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype('category')

    # Additional defensive check: if any categorical column has zero observed levels, bail out
    for cat_col in ['device', 'language', 'page_id']:
        if cat_col in df.columns:
            if df[cat_col].nunique(dropna=True) == 0:
                return None

    # Build formula (preserve exact variable names)
    formula = (
        'log_speed ~ reader_view * dyslexia_bin + age + C(device) + C(language) '
        '+ english_native + num_words + Flesch_Kincaid + correct_rate + C(page_id)'
    )

    # Fit OLS with clustered standard errors by uuid if uuid present
    try:
        if 'uuid' in df.columns and df['uuid'].nunique() > 1:
            model_fit = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['uuid']})
        else:
            # If there is only one cluster or no uuid variation, fall back to ordinary OLS fit
            model_fit = smf.ols(formula, data=df).fit()
    except Exception:
        # If fitting fails for any reason (e.g., perfect collinearity), return None
        return None

    return model_fit