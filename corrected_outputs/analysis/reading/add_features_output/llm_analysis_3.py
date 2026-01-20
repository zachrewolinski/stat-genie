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
    Transform the raw dataset into an analysis-ready dataframe for modeling the effect of Reader View
    on reading speed, and its moderation by dyslexia.

    Steps:
    - Make a copy of df
    - Drop retake trials and rows missing essential columns
    - Compute wpm from num_words and adjusted_running_time (ms -> minutes)
    - Filter trials with very small/non-positive times or low comprehension (correct_rate < 0.5)
    - Map english_native to binary
    - Create device dummies and ensure desktop/tablet columns exist (smartphone as reference)
    - Winsorize wpm at 1st/99th percentiles
    - Return dataframe with the exact columns used in modeling
    """
    df = df.copy()

    # Required columns for our transformations
    required_cols = [
        'adjusted_running_time', 'num_words', 'reader_view', 'dyslexia_bin',
        'retake_trial', 'correct_rate', 'english_native', 'device', 'age', 'Flesch_Kincaid'
    ]

    # Drop rows missing key columns
    df = df.dropna(subset=required_cols)

    # Exclude retake trials (we want first, non-retake trials)
    df = df[df['retake_trial'] == 0]

    # Ensure numeric types where expected
    df['adjusted_running_time'] = pd.to_numeric(df['adjusted_running_time'], errors='coerce')
    df['num_words'] = pd.to_numeric(df['num_words'], errors='coerce')
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce').fillna(0).astype(int)
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce').fillna(0).astype(int)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['Flesch_Kincaid'] = pd.to_numeric(df['Flesch_Kincaid'], errors='coerce')

    # Drop rows with non-positive or missing adjusted_runtime or num_words
    df = df.dropna(subset=['adjusted_running_time', 'num_words', 'age', 'Flesch_Kincaid'])
    df = df[df['adjusted_running_time'] > 200]  # exclude extremely small times (ms)

    # Compute words-per-minute (wpm). adjusted_running_time is in milliseconds.
    df['wpm'] = df['num_words'] / (df['adjusted_running_time'] / 60000.0)

    # Filter for minimum comprehension to ensure participant read the text; adjust threshold if desired
    df = df[df['correct_rate'] >= 0.5]

    # Map english_native to binary indicator (1 if 'Y', else 0)
    df['english_native_Y'] = df['english_native'].apply(lambda x: 1 if str(x).upper() == 'Y' else 0)

    # Create device dummy variables. We'll use smartphone as the reference and include desktop/tablet dummies.
    device_dummies = pd.get_dummies(df['device'].astype(str).str.lower(), prefix='device')
    # Normalize expected names in case of different capitalization/spaces
    # Common categories in the dataset: 'smartphone', 'tablet', 'desktop'
    # Ensure columns device_desktop and device_tablet exist; if not, create them filled with 0s
    for col in ['device_desktop', 'device_tablet', 'device_smartphone']:
        if col not in device_dummies.columns:
            device_dummies[col] = 0

    # Attach selected dummies to df
    df = pd.concat([df, device_dummies[['device_desktop', 'device_tablet']]], axis=1)

    # Winsorize wpm at 1st and 99th percentiles to reduce extreme outlier influence
    lower = np.percentile(df['wpm'].dropna(), 1)
    upper = np.percentile(df['wpm'].dropna(), 99)
    df['wpm'] = df['wpm'].clip(lower=lower, upper=upper)

    # Select final columns required for modeling and drop any remaining NA
    final_cols = [
        'wpm', 'reader_view', 'dyslexia_bin', 'age', 'num_words', 'Flesch_Kincaid',
        'english_native_Y', 'device_desktop', 'device_tablet'
    ]
    df = df[final_cols].dropna()

    # Ensure correct dtypes
    df['reader_view'] = df['reader_view'].astype(int)
    df['dyslexia_bin'] = df['dyslexia_bin'].astype(int)
    df['device_desktop'] = df['device_desktop'].astype(int)
    df['device_tablet'] = df['device_tablet'].astype(int)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit a linear model testing whether Reader View affects reading speed (wpm), and whether that effect
    differs for readers with dyslexia. Returns robust (HC3) standard-error regression results.

    Model specification:
    wpm ~ reader_view * dyslexia_bin + age + num_words + Flesch_Kincaid + english_native_Y + device_desktop + device_tablet

    The interaction term reader_view:dyslexia_bin tests whether the Reader View effect differs for
    readers with dyslexia (dyslexia_bin == 1) compared to non-dyslexic readers (dyslexia_bin == 0).
    """
    import statsmodels.formula.api as smf

    # Ensure the dataframe contains required columns
    required = ['wpm', 'reader_view', 'dyslexia_bin', 'age', 'num_words', 'Flesch_Kincaid',
                'english_native_Y', 'device_desktop', 'device_tablet']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns for modeling: {missing}')

    # Construct formula
    formula = 'wpm ~ reader_view * dyslexia_bin + age + num_words + Flesch_Kincaid + english_native_Y + device_desktop + device_tablet'

    # Fit OLS
    ols_mod = smf.ols(formula=formula, data=df).fit()

    # Convert to robust covariance results (HC3) for more reliable SEs in presence of heteroskedasticity
    robust_res = ols_mod.get_robustcov_results(cov_type='HC3')

    # Print a short summary for quick inspection (caller can inspect returned object for full details)
    print(robust_res.summary())

    return robust_res


