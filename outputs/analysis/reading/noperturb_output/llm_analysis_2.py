from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid side-effects
    df = df.copy()

    # Drop rows missing essential identifiers and treatment/diagnosis variables
    df = df.dropna(subset=['uuid', 'reader_view', 'dyslexia_bin', 'speed'])

    # Coerce numeric columns (if they come in as strings)
    numeric_cols = [
        'reader_view', 'dyslexia_bin', 'speed', 'age', 'num_words', 'Flesch_Kincaid',
        'correct_rate', 'retake_trial', 'img_width'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # After coercion, drop rows missing any numeric control required for the model
    needed_after_coercion = ['speed', 'reader_view', 'dyslexia_bin', 'num_words', 'Flesch_Kincaid', 'age']
    df = df.dropna(subset=needed_after_coercion)

    # Ensure binary encodings are integers
    df['reader_view'] = df['reader_view'].astype(int)
    df['dyslexia_bin'] = df['dyslexia_bin'].astype(int)
    # Handle retake_trial whether present or not
    if 'retake_trial' in df.columns:
        df['retake_trial'] = df['retake_trial'].fillna(0).astype(int)
    else:
        df['retake_trial'] = 0

    # Create dependent variable: log-transformed speed to stabilize variance and reduce skew
    # Add a small constant to avoid log(0)
    df['log_speed'] = np.log(df['speed'].clip(lower=0) + 1)

    # Center continuous covariates to improve interpretability of main effects
    df['age_c'] = df['age'] - df['age'].mean()
    df['num_words_c'] = df['num_words'] - df['num_words'].mean()
    df['Flesch_c'] = df['Flesch_Kincaid'] - df['Flesch_Kincaid'].mean()
    # img_width may be missing; handle gracefully
    if 'img_width' in df.columns:
        df['img_width_c'] = df['img_width'] - df['img_width'].mean()
    else:
        df['img_width_c'] = 0.0

    # Map english_native to binary 1/0 if present
    if 'english_native' in df.columns:
        df['english_native_bin'] = (
            df['english_native'].astype(str).str.upper().map({'Y': 1, 'N': 0})
        )
        # If there are other values or missing, treat them as 0 (not native)
        df['english_native_bin'] = df['english_native_bin'].fillna(0).astype(int)
    else:
        df['english_native_bin'] = 0

    # Coerce categorical columns to appropriate dtype for modeling (we will use C(...) in formula)
    for c in ['device', 'gender', 'page_id', 'uuid']:
        if c in df.columns:
            df[c] = df[c].astype('category')

    # Ensure required categorical columns exist in final dataframe (create defaults if absent)
    if 'device' not in df.columns:
        df['device'] = pd.Categorical(['missing'] * len(df))
    if 'gender' not in df.columns:
        df['gender'] = pd.Categorical(['missing'] * len(df))
    if 'page_id' not in df.columns:
        df['page_id'] = pd.Categorical(['missing'] * len(df))

    # Ensure correct_rate exists; if missing, fill with 0 (no correct responses)
    if 'correct_rate' not in df.columns:
        df['correct_rate'] = 0.0
    else:
        df['correct_rate'] = df['correct_rate'].fillna(0.0)

    # Ensure uuid is present and of appropriate type
    df['uuid'] = df['uuid'].astype(str).astype('category')

    # Drop rows with missing dependent variable
    df = df.dropna(subset=['log_speed'])

    # Reset index to ensure compatibility with statsmodels grouping internals
    df = df.reset_index(drop=True)

    # Return dataframe with columns needed for modeling (may contain other columns as well)
    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a mixed-effects linear model testing whether Reader View (reader_view)
    improves reading speed for individuals with dyslexia (dyslexia_bin). We
    include an interaction reader_view * dyslexia_bin to test differential effects
    and include participant random intercepts to account for repeated measures.

    Model specification (formula):
      log_speed ~ reader_view * dyslexia_bin + age_c + num_words_c + Flesch_c +
                  correct_rate + retake_trial + english_native_bin + img_width_c +
                  C(device) + C(gender) + C(page_id)

    Random effects: random intercept for uuid (groups=df['uuid']).

    Returns
    -------
    results : statsmodels mixedlm results object
    """
    # Work on a copy and ensure integer/ categorical types are appropriate
    df = df.copy().reset_index(drop=True)

    # Ensure required columns exist
    required = [
        'log_speed', 'reader_view', 'dyslexia_bin', 'uuid', 'age_c', 'num_words_c',
        'Flesch_c', 'correct_rate', 'retake_trial', 'english_native_bin', 'img_width_c',
        'device', 'gender', 'page_id'
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column {c} not present in dataframe")

    # Drop any rows with missing values in the required columns to avoid mismatches
    df = df.dropna(subset=required).reset_index(drop=True)

    # Ensure categorical columns are treated as categorical for the model
    for c in ['device', 'gender', 'page_id', 'uuid']:
        if not pd.api.types.is_categorical_dtype(df[c]):
            df[c] = df[c].astype('category')

    # Ensure reader_view and dyslexia_bin are numeric (0/1)
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce').fillna(0).astype(int)
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce').fillna(0).astype(int)

    # Define formula with interaction between reader_view and dyslexia_bin
    formula = (
        'log_speed ~ reader_view * dyslexia_bin + age_c + num_words_c + Flesch_c + '
        'correct_rate + retake_trial + english_native_bin + img_width_c + '
        'C(device) + C(gender) + C(page_id)'
    )

    # Ensure DataFrame index is a simple RangeIndex (some statsmodels internals expect positional indices)
    df.index = pd.RangeIndex(start=0, stop=len(df), step=1)

    # Use string labels for groups to avoid categorical-internal code issues
    groups = df['uuid'].astype(str).values

    # Fit mixed effects model with random intercept for participant (uuid)
    # Use REML=False for easier comparison with frequentist fixed-effect output
    md = smf.mixedlm(formula, data=df, groups=groups, re_formula='1')
    try:
        mdf = md.fit(reml=False, method='lbfgs')
    except Exception:
        # fallback to default method if convergence issues
        mdf = md.fit(reml=False)

    # Print summary to help interpretation in interactive use
    print(mdf.summary())

    return mdf