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
    Transform the raw dataset into a dataframe ready for modeling.

    New columns produced (and used by the model):
      - log_speed: natural log of speed (with small epsilon to avoid log(0))
      - reader_view: ensured integer 0/1
      - dyslexia_bin: ensured integer 0/1 (treats existing dyslexia_bin as authoritative)
      - english_native_bin: mapped from english_native ('Y'->1, 'N'->0)
      - num_words_z, Flesch_z, age_z, img_width_z: standardized continuous covariates
      - device, page_id, uuid: ensured categorical types

    The function drops rows missing the crucial variables for the analysis.
    """
    df = df.copy()

    # Required columns for the analysis
    required_cols = ['speed', 'reader_view', 'dyslexia_bin', 'uuid', 'page_id']
    # Drop rows missing essential columns
    df = df.dropna(subset=required_cols)

    # Ensure binary treatment and moderator are integer 0/1
    # reader_view is already 0/1 in schema, but coerce to int
    df['reader_view'] = df['reader_view'].astype(int)

    # dyslexia_bin in schema is 0/1; if there are other representations, coerce
    df['dyslexia_bin'] = df['dyslexia_bin'].astype(int)

    # Create dependent variable: log speed (add small epsilon to avoid log(0))
    eps = 1e-6
    df['log_speed'] = np.log(df['speed'].astype(float) + eps)

    # Map english_native to binary (Y -> 1, N -> 0). If missing, set to 0.
    if 'english_native' in df.columns:
        df['english_native_bin'] = df['english_native'].map({'Y': 1, 'N': 0})
        df['english_native_bin'] = df['english_native_bin'].fillna(0).astype(int)
    else:
        # If column absent, create default column of zeros
        df['english_native_bin'] = 0

    # Ensure retake_trial is numeric 0/1 if present
    if 'retake_trial' in df.columns:
        df['retake_trial'] = df['retake_trial'].fillna(0).astype(int)
    else:
        df['retake_trial'] = 0

    # Standardize continuous controls (z-score). Fill missing with median before standardizing.
    def zscore(col):
        return (col - col.mean()) / (col.std(ddof=0) if col.std(ddof=0) != 0 else 1)

    # num_words
    if 'num_words' in df.columns:
        df['num_words'] = df['num_words'].fillna(df['num_words'].median())
        df['num_words_z'] = zscore(df['num_words'].astype(float))
    else:
        df['num_words_z'] = 0.0

    # Flesch-Kincaid readability
    if 'Flesch_Kincaid' in df.columns:
        df['Flesch_Kincaid'] = df['Flesch_Kincaid'].fillna(df['Flesch_Kincaid'].median())
        df['Flesch_z'] = zscore(df['Flesch_Kincaid'].astype(float))
    else:
        df['Flesch_z'] = 0.0

    # Age
    if 'age' in df.columns:
        df['age'] = df['age'].fillna(df['age'].median())
        df['age_z'] = zscore(df['age'].astype(float))
    else:
        df['age_z'] = 0.0

    # img_width
    if 'img_width' in df.columns:
        df['img_width'] = df['img_width'].fillna(df['img_width'].median())
        df['img_width_z'] = zscore(df['img_width'].astype(float))
    else:
        df['img_width_z'] = 0.0

    # Ensure categorical columns are of type category
    for cat_col in ['device', 'page_id', 'uuid']:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype('category')
        else:
            # create a placeholder category if absent (shouldn't be absent for page_id/uuid per schema)
            df[cat_col] = pd.Categorical(['missing'] * len(df))

    # Final check: drop any remaining rows with NA in model columns
    model_cols = ['log_speed', 'reader_view', 'dyslexia_bin', 'num_words_z', 'Flesch_z', 'age_z', 'img_width_z', 'english_native_bin', 'retake_trial', 'device', 'page_id', 'uuid']
    df = df.dropna(subset=model_cols)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model testing whether Reader View improves reading speed for individuals with dyslexia.

    Model specification (primary):
      log_speed ~ reader_view * dyslexia_bin + num_words_z + Flesch_z + age_z + img_width_z
                   + english_native_bin + retake_trial + C(device) + C(page_id)

    - Interaction reader_view * dyslexia_bin tests whether the effect of Reader View differs
      between readers with and without dyslexia.
    - Page fixed effects (C(page_id)) control for page-specific difficulty/layout.
    - Cluster-robust standard errors are computed at the participant level (uuid).

    Returns the fitted statsmodels results object.
    """
    import statsmodels.formula.api as smf

    # Ensure transformed columns exist
    required = ['log_speed', 'reader_view', 'dyslexia_bin', 'uuid', 'page_id', 'device']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    formula = ('log_speed ~ reader_view * dyslexia_bin + num_words_z + Flesch_z + age_z + '
               'img_width_z + english_native_bin + retake_trial + C(device) + C(page_id)')

    model = smf.ols(formula=formula, data=df)

    # Fit with cluster-robust SEs at participant level (uuid). If uuid has many singletons,
    # fallback to heteroskedasticity-robust (HC1) standard errors.
    try:
        results = model.fit(cov_type='cluster', cov_kwds={'groups': df['uuid']})
    except Exception:
        # fallback
        results = model.fit(cov_type='HC1')

    # Print summary to help immediate inspection (caller can still use returned results)
    print(results.summary())

    return results


