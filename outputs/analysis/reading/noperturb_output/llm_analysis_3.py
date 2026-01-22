from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrices


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset to the analysis-ready dataframe.

    Steps:
    - copy dataframe to avoid in-place modifications
    - drop rows missing the core variables (speed, reader_view, dyslexia_bin, uuid, page_id)
    - remove non-positive speed values
    - coerce types for binary columns
    - create log_speed = log(speed)
    - coerce categorical columns to category dtype
    - impute (median) occasional missing control variables
    - return a dataframe that contains exactly the columns used in the model
    """
    df = df.copy()

    # Drop rows missing the essentials
    required = ['speed', 'reader_view', 'dyslexia_bin', 'uuid', 'page_id']
    df = df.dropna(subset=required)

    # Remove impossible/non-positive speed values
    df = df[df['speed'] > 0]

    # Ensure binary columns are integer 0/1 (if present)
    # Use fillna and map if values are booleans or strings representing 0/1
    # If values are non-numeric (e.g., 'Y'/'N'), attempt mapping; otherwise coerce to int after fillna.
    def ensure_binary(col_series):
        if col_series.dtype == bool:
            return col_series.astype(int)
        # Map common string encodings
        mapped = col_series.map({'Y': 1, 'N': 0, 'y': 1, 'n': 0, 'yes': 1, 'no': 0}).where(col_series.notna())
        # If mapping produced any non-nulls, use it; otherwise fallback to numeric coercion
        if mapped.notna().any():
            return mapped.fillna(0).astype(int)
        else:
            return col_series.fillna(0).astype(int)

    df['reader_view'] = ensure_binary(df['reader_view'])
    df['dyslexia_bin'] = ensure_binary(df['dyslexia_bin'])
    if 'retake_trial' in df.columns:
        df['retake_trial'] = ensure_binary(df['retake_trial'])

    # Create dependent variable: log-transformed speed
    df['log_speed'] = np.log(df['speed'])

    # Coerce categorical variables
    if 'device' in df.columns:
        # Replace missing device entries with explicit 'missing' category to avoid downstream NA drops
        df['device'] = df['device'].fillna('missing').astype('category')
    if 'english_native' in df.columns:
        df['english_native'] = df['english_native'].fillna('missing').astype('category')
    df['page_id'] = df['page_id'].astype('category')
    df['uuid'] = df['uuid'].astype('category')

    # Impute (median) for continuous controls if occasionally missing
    continuous_impute = ['age', 'num_words', 'Flesch_Kincaid', 'correct_rate', 'img_width']
    for col in continuous_impute:
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

    # Keep only the columns required for modeling (explicit list ensures exact column names are present)
    keep_cols = [
        'uuid', 'page_id', 'log_speed', 'reader_view', 'dyslexia_bin',
        'age', 'num_words', 'Flesch_Kincaid', 'retake_trial', 'correct_rate',
        'img_width', 'device', 'english_native'
    ]

    # If some optional control columns were not present in the raw df, drop them from the keep list gracefully
    keep_cols = [c for c in keep_cols if c in df.columns]

    # Return final dataframe with only the required columns for the model
    return df[keep_cols]


def model(df: pd.DataFrame) -> Any:
    """
    Fit a linear mixed effects model to estimate whether Reader View affects reading speed differently
    for readers with dyslexia.

    Model specification:
    - Dependent variable: log_speed
    - Key terms: reader_view, dyslexia_bin, and their interaction reader_view:dyslexia_bin
    - Controls: age, num_words, Flesch_Kincaid, retake_trial, correct_rate, img_width, device, english_native, page_id
    - Random intercept for participant (uuid) to account for repeated measures

    Returns the fitted MixedLMResults object.
    """
    # Ensure the columns used in the formula are present
    required = ['log_speed', 'reader_view', 'dyslexia_bin', 'uuid', 'page_id']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in dataframe passed to model().")

    # Work on a fresh copy and reset the index to ensure grouping logic works correctly
    df = df.copy().reset_index(drop=True)

    # Ensure uuid is categorical for clarity
    df['uuid'] = df['uuid'].astype('category')

    # Build formula dynamically so optional controls are only included if present
    formula_parts = ['log_speed ~ reader_view * dyslexia_bin']

    # Add continuous controls if present
    continuous_controls = ['age', 'num_words', 'Flesch_Kincaid', 'retake_trial', 'correct_rate', 'img_width']
    for col in continuous_controls:
        if col in df.columns:
            formula_parts.append(col)

    # Add categorical controls if present
    if 'device' in df.columns:
        formula_parts.append('C(device)')
    if 'english_native' in df.columns:
        formula_parts.append('C(english_native)')
    if 'page_id' in df.columns:
        formula_parts.append('C(page_id)')

    formula = ' + '.join(formula_parts)

    # Use patsy to construct design matrices so we can align the groups vector with any dropped rows
    # This avoids index misalignment errors when patsy drops rows with missing values.
    y, X = dmatrices(formula, df, return_type='dataframe')

    # Align groups to the rows retained in the design matrix
    groups = df.loc[X.index, 'uuid']

    # Fit mixed effects model with a random intercept per participant (uuid)
    md = sm.MixedLM(endog=y.iloc[:, 0], exog=X, groups=groups)

    # Use maximum likelihood (reml=False) so that comparisons are consistent if needed
    mdf = md.fit(reml=False, method='lbfgs', maxiter=200)

    # Print a compact summary and return the fitted model object
    print(mdf.summary())
    return mdf