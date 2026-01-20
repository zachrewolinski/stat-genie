from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/reading/replace_with_rvs_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Ensure participant id exists
    if 'uuid' not in df.columns:
        raise ValueError('Column uuid (participant id) is required for the analysis.')

    # Ensure dyslexia_bin exists; if only dyslexia present, derive dyslexia_bin (any dyslexia > 0 -> 1)
    if 'dyslexia_bin' not in df.columns and 'dyslexia' in df.columns:
        df['dyslexia_bin'] = (df['dyslexia'] > 0).astype(int)

    # Cast reader_view to int (0/1)
    if 'reader_view' in df.columns:
        df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce').astype('float').fillna(0).astype(int)

    # Create binary English native indicator if available
    if 'english_native' in df.columns:
        df['english_native_Y'] = df['english_native'].astype(str).str.upper().eq('Y').astype(int)
    else:
        df['english_native_Y'] = 0

    # Drop rows missing the essential modeling variables
    needed = ['speed', 'reader_view', 'dyslexia_bin', 'uuid']
    df = df.dropna(subset=needed)

    # Remove non-positive speed values (can't log-transform)
    df = df[df['speed'] > 0].copy()

    # Log-transform the dependent variable to reduce skew
    df['log_speed'] = np.log(df['speed'])

    # Ensure numeric controls exist and fill missing with column means where appropriate
    for col in ['age', 'num_words', 'Flesch_Kincaid', 'img_width', 'correct_rate', 'retake_trial']:
        if col not in df.columns:
            # If a control is missing entirely, create a default column with NaNs so later dropna will catch it
            df[col] = np.nan

    # Fill retake_trial and correct_rate missing values with 0 where reasonable
    if 'retake_trial' in df.columns:
        df['retake_trial'] = pd.to_numeric(df['retake_trial'], errors='coerce').fillna(0).astype(int)

    if 'correct_rate' in df.columns:
        df['correct_rate'] = pd.to_numeric(df['correct_rate'], errors='coerce').fillna(df['correct_rate'].mean())

    # Standardize continuous controls (z-score). Use population std (ddof=0) for scaling consistency
    def zscore(col):
        col_numeric = pd.to_numeric(df[col], errors='coerce')
        mean = col_numeric.mean()
        std = col_numeric.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=df.index)
        return (col_numeric - mean) / std

    df['age_z'] = zscore('age')
    df['num_words_z'] = zscore('num_words')
    df['Flesch_Kincaid_z'] = zscore('Flesch_Kincaid')
    df['img_width_z'] = zscore('img_width')

    # Keep device as categorical (ensure it's string/categorical dtype)
    if 'device' in df.columns:
        df['device'] = df['device'].astype('category')
    else:
        # create a default device category if missing
        df['device'] = 'unknown'
        df['device'] = df['device'].astype('category')

    # Select only rows with no missing values in the final modeling columns
    model_cols = [
        'uuid', 'log_speed', 'reader_view', 'dyslexia_bin', 'age_z', 'num_words_z',
        'Flesch_Kincaid_z', 'img_width_z', 'retake_trial', 'correct_rate', 'english_native_Y', 'device'
    ]
    df = df.dropna(subset=model_cols)

    # Return only the columns needed for modeling (plus any you want to inspect)
    return df[model_cols]


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Fit a linear mixed-effects model testing the interaction between reader_view and dyslexia
    # We model log_speed as the outcome, include participant random intercepts (groups=uuid),
    # and control for age, text length, readability, image width, retake flag, comprehension, english native, and device.

    import statsmodels.formula.api as smf

    # Build formula: interaction between reader_view and dyslexia_bin
    formula = (
        'log_speed ~ reader_view * dyslexia_bin '
        '+ age_z + num_words_z + Flesch_Kincaid_z + img_width_z '
        '+ retake_trial + correct_rate + english_native_Y + C(device)'
    )

    # Fit mixed linear model with participant-level random intercepts
    # Use reml=False to make results comparable to OLS-like inference
    results = smf.mixedlm(formula, data=df, groups=df['uuid']).fit(reml=False)

    # Return the fitted results object (has .summary(), .params, .pvalues, etc.)
    return results


