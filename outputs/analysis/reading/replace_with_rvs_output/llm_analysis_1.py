from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/reading/replace_with_rvs_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw dataset into the analysis dataframe.
    Produces the following new/clean columns used in modeling:
      - log_speed: np.log(speed + 1)
      - reader_view: ensured integer 0/1
      - dyslexia_bin: ensured integer 0/1 (already present in dataset as 'dyslexia_bin')
      - english_native_bin: 1 if english_native == 'Y' else 0
    Also fills or drops missing values required by the model and ensures types.
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure key columns exist
    required_cols = ['speed', 'reader_view', 'dyslexia_bin', 'uuid', 'device', 'num_words',
                     'Flesch_Kincaid', 'age', 'english_native', 'retake_trial', 'correct_rate']

    # If some optional controls are missing, add them as NaN so downstream code can handle
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Drop rows with missing essential variables: speed, reader_view, dyslexia_bin, uuid
    df = df.dropna(subset=['speed', 'reader_view', 'dyslexia_bin', 'uuid'])

    # Ensure numeric types
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce').fillna(0).astype(int)
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce')

    # Drop rows where dyslexia_bin is missing after coercion
    df = df.dropna(subset=['dyslexia_bin'])
    df['dyslexia_bin'] = df['dyslexia_bin'].astype(int)

    # Create log-transformed speed to reduce skew. Add a small offset to avoid log(0).
    df['log_speed'] = np.log(df['speed'].astype(float) + 1.0)

    # english_native -> binary
    if 'english_native' in df.columns:
        # common entries are 'Y'/'N'; treat anything starting with 'Y' or 'y' as 1
        df['english_native_bin'] = df['english_native'].astype(str).str.upper().str.startswith('Y').astype(int)
    else:
        df['english_native_bin'] = 0

    # Ensure numeric controls and impute simple missing values with median where reasonable
    numeric_controls = ['num_words', 'Flesch_Kincaid', 'age', 'retake_trial', 'correct_rate']
    for col in numeric_controls:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # impute median for controls (so we keep rows rather than drop many)
            median_val = df[col].median()
            if pd.isna(median_val):
                # if column entirely NaN, fill with 0
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(median_val)
        else:
            df[col] = 0

    # Device: keep as categorical; replace missing with 'unknown'
    if 'device' in df.columns:
        df['device'] = df['device'].fillna('unknown').astype(str)
    else:
        df['device'] = 'unknown'

    # uuid: ensure string type for clustering
    df['uuid'] = df['uuid'].astype(str)

    # Final: drop any rows where log_speed is infinite or NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['log_speed'])

    # Keep only columns needed downstream to keep dataframe compact (but don't drop any we declared)
    keep_cols = ['uuid', 'reader_view', 'dyslexia_bin', 'log_speed', 'num_words', 'Flesch_Kincaid',
                 'age', 'device', 'english_native_bin', 'retake_trial', 'correct_rate', 'speed']
    existing_keep = [c for c in keep_cols if c in df.columns]
    df = df[existing_keep].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a linear model testing whether Reader View (reader_view) improves reading speed
    and whether that effect differs for individuals with dyslexia (dyslexia_bin).

    Model specification (primary):
      log_speed ~ reader_view * dyslexia_bin + num_words + Flesch_Kincaid + age
                  + C(device) + english_native_bin + retake_trial + correct_rate

    We use OLS with participant-level clustered standard errors (cluster by uuid) to
    account for repeated measures per participant. Returns the fitted results object.
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # Ensure the dataframe contains the columns the model expects
    required = ['log_speed', 'reader_view', 'dyslexia_bin', 'num_words', 'Flesch_Kincaid',
                'age', 'device', 'english_native_bin', 'retake_trial', 'correct_rate', 'uuid']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    formula = ('log_speed ~ reader_view * dyslexia_bin + num_words + Flesch_Kincaid + age '
               '+ C(device) + english_native_bin + retake_trial + correct_rate')

    # Fit OLS
    ols_mod = smf.ols(formula, data=df)
    # Clustered standard errors by participant uuid
    results = ols_mod.fit(cov_type='cluster', cov_kwds={'groups': df['uuid']})

    # Print summary for immediate inspection (caller can also use returned object)
    print(results.summary())

    return results


