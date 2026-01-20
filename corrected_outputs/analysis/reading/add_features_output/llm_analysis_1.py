from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/reading/add_features_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for modeling. The function:
      - Drops rows missing core variables (speed, reader_view, dyslexia_bin, uuid).
      - Ensures types for key variables.
      - Creates log-transformed speed (speed_log).
      - Creates centered age (age_c) and binary english native indicator (english_native_bin).
      - Ensures categorical variables are typed appropriately.

    Final dataframe includes the exact columns referenced in the model:
      - speed_log, reader_view, dyslexia_bin, num_words, Flesch_Kincaid,
        age_c, device, english_native_bin, retake_trial, page_id, uuid
    """
    df = df.copy()

    # Drop rows missing essential variables
    essential = ['speed', 'reader_view', 'dyslexia_bin', 'uuid']
    df = df.dropna(subset=essential)

    # Ensure numeric conversions where appropriate
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df = df.dropna(subset=['speed'])

    # Binary treatment and moderator
    # reader_view is expected 0/1 already, coerce to int
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce').astype(int)
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce').astype(int)

    # Dependent variable: log-transform speed to stabilize variance
    # add a tiny constant for numerical safety (speed > 0 in dataset)
    df['speed_log'] = np.log(df['speed'] + 1e-6)

    # Controls: ensure numeric
    if 'num_words' in df.columns:
        df['num_words'] = pd.to_numeric(df['num_words'], errors='coerce')
    else:
        df['num_words'] = np.nan

    if 'Flesch_Kincaid' in df.columns:
        df['Flesch_Kincaid'] = pd.to_numeric(df['Flesch_Kincaid'], errors='coerce')
    else:
        df['Flesch_Kincaid'] = np.nan

    # Age: numeric and centered
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        # center using available mean
        df['age_c'] = df['age'] - df['age'].mean()
    else:
        df['age_c'] = np.nan

    # English native: convert 'Y' -> 1, others -> 0 (handles missing as 0)
    if 'english_native' in df.columns:
        df['english_native_bin'] = df['english_native'].map({'Y': 1}).fillna(0).astype(int)
    else:
        df['english_native_bin'] = 0

    # Retake trial: ensure binary int
    if 'retake_trial' in df.columns:
        df['retake_trial'] = pd.to_numeric(df['retake_trial'], errors='coerce').fillna(0).astype(int)
    else:
        df['retake_trial'] = 0

    # Categorical variables
    if 'device' in df.columns:
        df['device'] = df['device'].astype('category')
    else:
        df['device'] = pd.Categorical(pd.Series([None] * len(df)))

    if 'page_id' in df.columns:
        df['page_id'] = df['page_id'].astype('category')
    else:
        df['page_id'] = pd.Categorical(pd.Series([None] * len(df)))

    # uuid kept as-is for clustering; ensure it's present
    df['uuid'] = df['uuid'].astype(str)

    # Keep only columns necessary for modeling to avoid accidental name mismatches
    keep_cols = [
        'speed_log', 'reader_view', 'dyslexia_bin', 'num_words', 'Flesch_Kincaid',
        'age_c', 'device', 'english_native_bin', 'retake_trial', 'page_id', 'uuid'
    ]

    # Some of these may not exist if upstream data lacked them; select safely
    existing_keep = [c for c in keep_cols if c in df.columns]
    df = df[existing_keep].copy()

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS regression to estimate the effect of Reader View on reading speed
    and whether that effect differs for readers with dyslexia.

    Model specification (primary):
        speed_log ~ reader_view * dyslexia_bin + num_words + Flesch_Kincaid + age_c
                     + english_native_bin + retake_trial + C(device) + C(page_id)

    We cluster standard errors by participant uuid to account for repeated measures.
    Returns the fitted model results object (with clustered robust SE).
    """
    import statsmodels.formula.api as smf

    # Ensure required columns present
    required = ['speed_log', 'reader_view', 'dyslexia_bin', 'uuid']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    # Build formula. Include categorical device and page_id if present.
    formula = (
        'speed_log ~ reader_view * dyslexia_bin + num_words + Flesch_Kincaid + age_c '
        '+ english_native_bin + retake_trial'
    )

    if 'device' in df.columns:
        formula += ' + C(device)'
    if 'page_id' in df.columns:
        formula += ' + C(page_id)'

    # Drop rows with any remaining NA in variables used by the formula
    # Patsy will complain on NA, so drop them explicitly
    used_vars = list(set([v.strip() for v in formula.replace('~', '+').split('+')]))
    # Remove 'C(device)' and 'C(page_id)' tokens to actual column names
    used_vars = [v.replace('C(', '').replace(')', '').strip() for v in used_vars if v.strip() != '']
    present_vars = [v for v in used_vars if v in df.columns]
    df_model = df.dropna(subset=present_vars).copy()

    # Fit OLS with clustered standard errors by uuid
    model = smf.ols(formula, data=df_model).fit(
        cov_type='cluster',
        cov_kwds={'groups': df_model['uuid']}
    )

    # Print a brief summary; return the fitted results for further inspection
    print(model.summary())
    return model


