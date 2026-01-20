from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/reading/shuffle_names_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying original
    df = df.copy()

    # Map/rename columns whose semantic descriptions correspond to reading and word counts
    # According to the provided schema: 'language' column contains reading time on page (ms) excluding scrolling
    # 'dyslexia' column contains the number of words on the page (word count)
    # 'running_time' indicates whether Reader View was activated (1/0)

    # Keep needed raw columns, drop rows with missing critical values
    required_cols = ['running_time', 'language', 'dyslexia', 'dyslexia_bin', 'device', 'age', 'gender', 'Flesch_Kincaid', 'img_width', 'retake_trial']
    for c in required_cols:
        if c not in df.columns:
            # create empty column if missing so pipeline fails later clearly
            df[c] = np.nan

    df = df.dropna(subset=['running_time', 'language', 'dyslexia', 'dyslexia_bin'])

    # Create clear, explicit columns used in modeling
    # Reading time in milliseconds (use 'language' field per dataset description)
    df['ReadingTimeMs'] = pd.to_numeric(df['language'], errors='coerce')
    # Words on page (use 'dyslexia' field per dataset description)
    df['WordsOnPage'] = pd.to_numeric(df['dyslexia'], errors='coerce')

    # Treatment indicator: Reader View (0/1) from 'running_time'
    df['ReaderView'] = pd.to_numeric(df['running_time'], errors='coerce').fillna(0).astype(int)

    # Dyslexia status: binary indicator for any dyslexia (1 or 2 in dyslexia_bin -> 1)
    df['DyslexiaAny'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce').fillna(0).astype(int)
    df['DyslexiaAny'] = (df['DyslexiaAny'] >= 1).astype(int)

    # Filter out non-positive times or word counts
    df = df[(df['ReadingTimeMs'] > 0) & (df['WordsOnPage'] > 0)]

    # Compute words per minute: WPM = words / (minutes) ; minutes = ms / 60000
    df['ReadingWPM'] = df['WordsOnPage'] * 60000.0 / df['ReadingTimeMs']

    # Drop any non-finite or zero WPM values (log will require positive values)
    df = df[np.isfinite(df['ReadingWPM']) & (df['ReadingWPM'] > 0)]

    # Log-transform WPM to reduce skew and make residuals more gaussian-like
    df['LogReadingWPM'] = np.log(df['ReadingWPM'])

    # Ensure control columns exist and have appropriate dtypes
    # device, age, gender, Flesch_Kincaid, img_width, retake_trial
    if 'device' in df.columns:
        # keep as-is (categorical levels will be handled in the model via C(device))
        df['device'] = df['device']
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    if 'gender' in df.columns:
        df['gender'] = pd.to_numeric(df['gender'], errors='coerce')
    if 'Flesch_Kincaid' in df.columns:
        df['Flesch_Kincaid'] = pd.to_numeric(df['Flesch_Kincaid'], errors='coerce')
    if 'img_width' in df.columns:
        df['img_width'] = pd.to_numeric(df['img_width'], errors='coerce')
    if 'retake_trial' in df.columns:
        df['retake_trial'] = df['retake_trial']

    # Keep only the columns necessary for modeling to make the returned dataframe compact
    keep_cols = ['LogReadingWPM', 'ReadingWPM', 'ReaderView', 'DyslexiaAny', 'device', 'age', 'gender', 'Flesch_Kincaid', 'img_width', 'retake_trial', 'ReadingTimeMs', 'WordsOnPage']
    df = df[keep_cols].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    import statsmodels.formula.api as smf

    # Expect df to be the transformed dataframe with required columns
    # Formula: main effect of ReaderView, interaction with DyslexiaAny (moderator), plus controls
    formula = (
        'LogReadingWPM ~ ReaderView * DyslexiaAny '
        '+ C(device) + age + gender + Flesch_Kincaid + img_width + C(retake_trial)'
    )

    # Fit OLS using formula API
    ols_mod = smf.ols(formula, data=df)
    ols_res = ols_mod.fit()

    # Provide robust standard errors (HC3) to protect against heteroskedasticity
    robust_res = ols_res.get_robustcov_results(cov_type='HC3')

    # Return the robust results object (has .summary(), .params, .bse, etc.)
    return robust_res


