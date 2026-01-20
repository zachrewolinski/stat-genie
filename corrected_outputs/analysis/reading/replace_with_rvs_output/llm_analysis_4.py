from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle


# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure key columns exist with sensible defaults so function always returns the required columns.
    # If the raw data truly lacks these, defaults will be used (and downstream model will drop rows with missing values).
    if 'reader_view' not in df.columns:
        df['reader_view'] = 0
    if 'dyslexia_bin' not in df.columns:
        df['dyslexia_bin'] = 0
    if 'speed' not in df.columns:
        df['speed'] = np.nan

    # Drop rows missing essential measurement values (speed, reader_view, dyslexia_bin)
    df = df.dropna(subset=['speed', 'reader_view', 'dyslexia_bin'])

    # Ensure speed is positive and drop non-positive speeds
    df = df[df['speed'] > 0]

    # Winsorize speed at 1st and 99th percentiles to reduce influence of extreme outliers
    if df['speed'].notna().any():
        p1 = np.percentile(df['speed'].dropna(), 1)
        p99 = np.percentile(df['speed'].dropna(), 99)
        df['speed_winsor'] = np.clip(df['speed'], p1, p99)
    else:
        df['speed_winsor'] = np.nan

    # Log transform the (winsorized) speed to reduce skew
    df['LogSpeed'] = np.log(df['speed_winsor'])

    # Ensure dyslexia_bin is integer 0/1
    # If dyslexia_bin encodes presence as 1, keep as is. Cast to int for modeling
    # Coerce to numeric first to avoid errors
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce').fillna(0).astype(int)

    # Center continuous controls for interpretability
    if 'num_words' in df.columns:
        df['num_words_c'] = df['num_words'] - df['num_words'].mean()
    else:
        df['num_words_c'] = 0.0

    if 'Flesch_Kincaid' in df.columns:
        df['Flesch_Kincaid_c'] = df['Flesch_Kincaid'] - df['Flesch_Kincaid'].mean()
    else:
        df['Flesch_Kincaid_c'] = 0.0

    if 'age' in df.columns:
        # treat missing ages conservatively by dropping rows with missing age (age is an important control)
        df = df.dropna(subset=['age'])
        df['age_c'] = df['age'] - df['age'].mean()
    else:
        df['age_c'] = 0.0

    # Ensure categorical variables are treated as categories.
    # If missing, create a single-category column 'missing' so the final dataframe has the required columns.
    for col in ['device', 'education', 'english_native', 'page_id']:
        if col in df.columns:
            df[col] = df[col].astype('category')
        else:
            df[col] = pd.Categorical(['missing'] * len(df))

    # retake_trial: ensure present and numeric
    if 'retake_trial' in df.columns:
        df['retake_trial'] = pd.to_numeric(df['retake_trial'], errors='coerce').fillna(0).astype(int)
    else:
        df['retake_trial'] = 0

    # Ensure uuid exists (used for clustering). If missing, create a per-row id.
    if 'uuid' not in df.columns:
        df['uuid'] = ['uid_{}'.format(i) for i in range(len(df))]

    # Keep only the columns required for modeling to simplify downstream code.
    required_cols = [
        'LogSpeed', 'reader_view', 'dyslexia_bin', 'num_words_c', 'Flesch_Kincaid_c', 'age_c',
        'retake_trial', 'device', 'education', 'english_native', 'page_id', 'uuid', 'speed', 'speed_winsor'
    ]

    # Ensure all required columns exist in df (create defaults if necessary)
    for c in required_cols:
        if c not in df.columns:
            # Assign sensible defaults depending on expected type
            if c in {'reader_view', 'dyslexia_bin', 'retake_trial'}:
                df[c] = 0
            elif c in {'num_words_c', 'Flesch_Kincaid_c', 'age_c', 'LogSpeed', 'speed', 'speed_winsor'}:
                df[c] = np.nan
            else:
                # categorical or uuid
                if c == 'uuid':
                    df[c] = ['uid_{}'.format(i) for i in range(len(df))]
                else:
                    df[c] = pd.Categorical(['missing'] * len(df))

    # Final selection and reset index
    df = df[required_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    import statsmodels.formula.api as smf

    # Formula: main effect of reader_view, dyslexia moderator, and their interaction.
    # Control for continuous covariates and include categorical fixed effects for device, education, english_native, and page_id.
    formula = (
        'LogSpeed ~ reader_view * dyslexia_bin '
        '+ num_words_c + Flesch_Kincaid_c + age_c + retake_trial '
        '+ C(device) + C(education) + C(english_native) + C(page_id)'
    )

    # Fit OLS
    fit = smf.ols(formula, data=df).fit()

    # Obtain cluster-robust standard errors clustered by participant (uuid) when uuid exists
    if 'uuid' in df.columns:
        try:
            results = fit.get_robustcov_results(cov_type='cluster', groups=df['uuid'])
        except Exception:
            # fallback to heteroskedasticity-robust (HC3)
            results = fit.get_robustcov_results(cov_type='HC3')
    else:
        results = fit

    # Return the fitted results object (has params, pvalues, summary(), etc.)
    return results