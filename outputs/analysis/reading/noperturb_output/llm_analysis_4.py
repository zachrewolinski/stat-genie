from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/reading/noperturb_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for modeling. Creates log-transformed speed (LogSpeed), ensures key variables are typed
    correctly, encodes English-native as binary english_native_Y, and drops rows with missing values in model
    variables. Also converts some columns to categorical for later use in formula (C(...)).

    Returned dataframe contains at least the following columns used by the model:
      - LogSpeed, reader_view, dyslexia_bin, age, num_words, Flesch_Kincaid, retake_trial,
        english_native_Y, device, page_id, uuid
    """
    df = df.copy()

    # Ensure main columns exist; if not, this will raise a KeyError which surfaces missing data problems early
    required_cols = [
        'speed', 'reader_view', 'dyslexia_bin', 'age', 'num_words', 'Flesch_Kincaid',
        'retake_trial', 'english_native', 'device', 'page_id', 'uuid'
    ]

    # Convert and clean basic types
    # reader_view and dyslexia_bin should be binary integers
    if 'reader_view' in df.columns:
        df['reader_view'] = df['reader_view'].astype(float)
        df['reader_view'] = df['reader_view'].fillna(0).astype(int)

    if 'dyslexia_bin' in df.columns:
        # datasetProvided dyslexia_bin should be 0/1; coerce to int
        df['dyslexia_bin'] = df['dyslexia_bin'].astype(float).fillna(0).astype(int)

    # Create english_native_Y binary: 'Y' -> 1, others -> 0. If column missing or different coding, this creates a conservative default (0).
    if 'english_native' in df.columns:
        df['english_native_Y'] = df['english_native'].map({'Y': 1, 'N': 0})
        # if any other string values occur, treat them as 0 (non-native)
        df['english_native_Y'] = df['english_native_Y'].fillna(0).astype(int)
    else:
        # If column missing entirely, create default column of zeros
        df['english_native_Y'] = 0

    # Log-transform speed to reduce skew (speed must be > 0)
    # Keep only rows with positive speed; drop non-positive or missing
    df = df[df['speed'] > 0].copy()
    df['LogSpeed'] = np.log(df['speed'].astype(float))

    # Convert categorical variables to category dtype to be handled by formula with C(...)
    if 'device' in df.columns:
        df['device'] = df['device'].astype('category')
    if 'page_id' in df.columns:
        df['page_id'] = df['page_id'].astype('category')
    if 'uuid' in df.columns:
        df['uuid'] = df['uuid'].astype('category')

    # Ensure numeric controls are numeric
    for col in ['age', 'num_words', 'Flesch_Kincaid', 'retake_trial']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values in any column that will be used in the model
    model_columns = [
        'LogSpeed', 'reader_view', 'dyslexia_bin', 'age', 'num_words', 'Flesch_Kincaid',
        'retake_trial', 'english_native_Y', 'device', 'page_id', 'uuid'
    ]
    df = df.dropna(subset=model_columns)

    # Final type enforcement
    df['reader_view'] = df['reader_view'].astype(int)
    df['dyslexia_bin'] = df['dyslexia_bin'].astype(int)
    df['retake_trial'] = df['retake_trial'].astype(int)

    # Return dataframe with all required columns (and any other original columns preserved)
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit a linear model to estimate the effect of Reader View on log reading speed and whether that effect differs
    for readers with dyslexia. Uses clustered standard errors at the reader (uuid) level.

    Model formula:
      LogSpeed ~ reader_view * dyslexia_bin + age + num_words + Flesch_Kincaid + retake_trial + english_native_Y
                   + C(device) + C(page_id)

    Returns:
      statsmodels regression results object (fitted model with cluster-robust SEs)
    """
    import statsmodels.formula.api as smf

    # Formula including interaction between reader_view and dyslexia_bin
    formula = (
        'LogSpeed ~ reader_view * dyslexia_bin + age + num_words + Flesch_Kincaid + '
        'retake_trial + english_native_Y + C(device) + C(page_id)'
    )

    # Fit OLS
    model = smf.ols(formula, data=df)

    # Cluster standard errors by reader (uuid) to account for repeated observations per participant
    # If uuid is categorical, use its raw values as groups
    try:
        results = model.fit(cov_type='cluster', cov_kwds={'groups': df['uuid']})
    except Exception:
        # Fallback to robust (HC3) if clustering fails for any reason
        results = model.fit(cov_type='HC3')

    # Print summary for quick inspection; return results for programmatic access
    print(results.summary())
    return results


