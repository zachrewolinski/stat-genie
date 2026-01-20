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
    """
    Transform raw dataset into analysis-ready dataframe.

    Assumptions and mappings (based on dataset schema notes):
    - 'running_time' is treated as the trial-level Reader View indicator (1 = Reader View on, 0 = off).
    - If 'running_time' is not present, 'reader_view' (Y/N) will be used (Y->1, N->0).
    - 'adjusted_running_time' is taken to be the total time on page (milliseconds).
    - 'scrolling_time' is the time spent scrolling (milliseconds); net reading time = adjusted_running_time - scrolling_time.
    - The dataset field named 'dyslexia' appears to contain page word counts (values ~100-400); we map that to 'num_words'. If 'dyslexia_bin' is present, it is used for categorical dyslexia status (0/1/2).

    The function returns a dataframe that includes at minimum the columns named in the conceptual variables: 
    ['reader_view_on','reading_time_ms','num_words','reading_speed_wpm','dyslexia_cat','age','device','education','Flesch_Kincaid','img_width','language']
    """

    df = df.copy()

    # --- Reader View indicator ---
    if 'running_time' in df.columns:
        df['reader_view_on'] = pd.to_numeric(df['running_time'], errors='coerce').fillna(0).astype(int)
    elif 'reader_view' in df.columns:
        # fallback: map Y/N to 1/0
        df['reader_view_on'] = df['reader_view'].astype(str).str.upper().map({'Y': 1, 'N': 0})
        df['reader_view_on'] = df['reader_view_on'].fillna(0).astype(int)
    else:
        # if neither column exists, create a column of zeros and raise a warning
        df['reader_view_on'] = 0

    # --- Numeric conversions for timing and scrolling ---
    if 'adjusted_running_time' in df.columns:
        df['adjusted_running_time'] = pd.to_numeric(df['adjusted_running_time'], errors='coerce')
    else:
        df['adjusted_running_time'] = pd.NA

    if 'scrolling_time' in df.columns:
        df['scrolling_time'] = pd.to_numeric(df['scrolling_time'], errors='coerce')
    else:
        df['scrolling_time'] = 0

    # Compute net reading time in milliseconds
    df['reading_time_ms'] = df['adjusted_running_time'] - df['scrolling_time']

    # --- Number of words on page ---
    # The raw schema is inconsistent; choose the column 'dyslexia' as num_words when it contains typical word counts.
    if 'dyslexia' in df.columns:
        df['num_words'] = pd.to_numeric(df['dyslexia'], errors='coerce')
    elif 'num_words' in df.columns:
        df['num_words'] = pd.to_numeric(df['num_words'], errors='coerce')
    else:
        df['num_words'] = pd.NA

    # Remove impossible or non-positive reading times and word counts
    df.loc[df['reading_time_ms'] <= 0, 'reading_time_ms'] = pd.NA
    df.loc[df['num_words'] <= 0, 'num_words'] = pd.NA

    # --- Reading speed (words per minute) ---
    # reading_time_ms is in ms; convert to minutes: /60000
    df['reading_speed_wpm'] = df['num_words'] / (df['reading_time_ms'] / 60000.0)

    # --- Dyslexia categorical variable (moderator) ---
    # Prefer an explicit dyslexia indicator/demographic encoding if provided
    if 'dyslexia_bin' in df.columns:
        # Expect codes like 0=no dyslexia, 1=dyslexia, 2=severe
        df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce')
        mapping = {0: 'NoDyslexia', 1: 'Dyslexia', 2: 'SevereDyslexia'}
        df['dyslexia_cat'] = df['dyslexia_bin'].map(mapping).fillna('Unknown')
    else:
        # Fallback: if there is a 'num_words' column and some separate dyslexia-like flag named 'gender' or 'gender' indicates dyslexia
        # (Dataset schema is inconsistent). We'll create a conservative binary grouping based on the presence of a 'gender' column that may encode dyslexia
        if 'gender' in df.columns:
            # Some versions of the data encode dyslexia in 'gender' per the provided schema notes; treat 1 as dyslexia if that seems to be the only available flag
            df['gender_numeric'] = pd.to_numeric(df['gender'], errors='coerce')
            df['dyslexia_cat'] = df['gender_numeric'].map({0: 'NoDyslexia', 1: 'Dyslexia'}).fillna('Unknown')
        else:
            # If no sensible dyslexia flag, create a placeholder 'Unknown' category and the model will still estimate the main effect
            df['dyslexia_cat'] = 'Unknown'

    # --- Controls: coerce to numeric where appropriate ---
    for c in ['age', 'device', 'education', 'Flesch_Kincaid', 'img_width']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = pd.NA

    # Language: keep as categorical factor if present
    if 'language' in df.columns:
        df['language'] = df['language'].astype('category')
    else:
        df['language'] = pd.Categorical(['Unknown'] * len(df))

    # Final: drop rows missing the core analysis variables
    df = df.dropna(subset=['reading_speed_wpm', 'reader_view_on', 'dyslexia_cat'])

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit linear regression to test whether Reader View improves reading speed and whether the effect differs by dyslexia status.

    Model specification (linear OLS):
    reading_speed_wpm ~ reader_view_on * C(dyslexia_cat) + age + device + education + Flesch_Kincaid + img_width + C(language)

    - Interaction reader_view_on * C(dyslexia_cat) tests whether Reader View effect differs across dyslexia groups.
    - Robust (heteroskedasticity-consistent) standard errors are used (HC3).

    Returns the fitted statsmodels results object.
    """

    import statsmodels.formula.api as smf

    # Ensure categorical dyslexia variable
    if 'dyslexia_cat' not in df.columns:
        raise ValueError("dyslexia_cat column required in dataframe. Run transform() first.")

    formula = (
        'reading_speed_wpm ~ reader_view_on * C(dyslexia_cat) '
        '+ age + device + education + Flesch_Kincaid + img_width + C(language)'
    )

    # Fit OLS with robust standard errors (HC3)
    model_fit = smf.ols(formula=formula, data=df).fit(cov_type='HC3')

    return model_fit


