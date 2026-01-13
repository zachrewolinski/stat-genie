from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/reading/add_features_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe.

    Steps:
    - Keep only non-retake (retake_trial == 0) trials to avoid learning/retest contamination.
    - Require non-missing adjusted_running_time and num_words and dyslexia_bin.
    - Filter out implausibly small adjusted_running_time (<= 0) and extremely short trials (<= 200 ms).
    - Compute ReadingWPM = (num_words / adjusted_running_time_ms) * 60000.
    - Winsorize ReadingWPM at 1st and 99th percentiles to limit influence of extreme outliers.
    - Map english_native to binary (1 for 'Y', 0 for 'N'); fill missing as 0.
    - Ensure dyslexia_bin is integer 0/1.
    - Cast device to category and reader_view to integer 0/1.

    Returns the dataframe containing all columns listed in the conceptual variables.
    """
    # Make a copy
    df = df.copy()

    # Ensure numeric columns exist and coerce types where appropriate
    df['adjusted_running_time'] = pd.to_numeric(df['adjusted_running_time'], errors='coerce')
    df['num_words'] = pd.to_numeric(df['num_words'], errors='coerce')
    # dyslexia_bin should already be 0/1 but coerce and fill
    df['dyslexia_bin'] = pd.to_numeric(df.get('dyslexia_bin', df.get('dyslexia')), errors='coerce')

    # Filter out retake trials (we focus on participants' primary readings)
    if 'retake_trial' in df.columns:
        df = df[df['retake_trial'] == 0]

    # Drop rows with critical missing values
    df = df.dropna(subset=['adjusted_running_time', 'num_words', 'dyslexia_bin'])

    # Remove non-positive or extremely small adjusted running times (likely faulty logs)
    df = df[df['adjusted_running_time'] > 200]

    # Compute reading speed in words per minute (WPM)
    df['ReadingWPM'] = df['num_words'] / df['adjusted_running_time'] * 60000.0

    # Winsorize ReadingWPM at 1st and 99th percentiles to reduce extreme outlier influence
    lower = df['ReadingWPM'].quantile(0.01)
    upper = df['ReadingWPM'].quantile(0.99)
    df['ReadingWPM'] = df['ReadingWPM'].clip(lower=lower, upper=upper)

    # Map english_native to binary
    if 'english_native' in df.columns:
        df['english_native'] = df['english_native'].map({
            'Y': 1,
            'N': 0
        }).fillna(0).astype(int)
    else:
        df['english_native'] = 0

    # Ensure dyslexia_bin is integer 0/1
    df['dyslexia_bin'] = df['dyslexia_bin'].astype(int)

    # Ensure reader_view is integer 0/1
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce').fillna(0).astype(int)

    # Ensure device is categorical
    if 'device' in df.columns:
        df['device'] = df['device'].astype('category')
    else:
        df['device'] = 'unknown'

    # Keep only rows with non-missing key controls used in the model
    model_needed = ['age', 'Flesch_Kincaid', 'correct_rate']
    for col in model_needed:
        if col not in df.columns:
            df[col] = np.nan

    # Optionally drop rows with missing outcome or key covariates
    df = df.dropna(subset=['ReadingWPM', 'dyslexia_bin', 'age', 'Flesch_Kincaid', 'correct_rate'])

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a mixed-effects linear model predicting ReadingWPM with Reader View, Dyslexia (moderator),
    their interaction, and covariates. We include a random intercept per participant (uuid) to account
    for repeated measures.

    Model formula:
      ReadingWPM ~ reader_view * dyslexia_bin + age + C(device) + english_native + Flesch_Kincaid + num_words + correct_rate

    Returns the fitted mixed model results object.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns are present
    required = ['ReadingWPM', 'reader_view', 'dyslexia_bin', 'age', 'device', 'english_native', 'Flesch_Kincaid', 'num_words', 'correct_rate', 'uuid']
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Dataframe missing required columns for modeling: {missing}")

    # Drop any remaining rows with NA in model vars
    df_model = df.dropna(subset=required)

    # Formula with interaction between reader_view and dyslexia_bin
    formula = 'ReadingWPM ~ reader_view * dyslexia_bin + age + C(device) + english_native + Flesch_Kincaid + num_words + correct_rate'

    # Fit mixed effects model: random intercept for participant (uuid)
    md = smf.mixedlm(formula, df_model, groups=df_model['uuid'], re_formula='~1')
    mdf = md.fit(reml=False, method='lbfgs')

    # Print summary for quick inspection
    print(mdf.summary())

    return mdf


