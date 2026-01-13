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
    Transform raw dataset into a dataframe suitable for modeling.

    Produces the following additional/modified columns required by the model:
      - log_speed: natural log of speed (dependent variable)
      - log_num_words: natural log of num_words (control)
      - english_native_binary: 1 if english_native == 'Y', else 0

    Filters rows with missing or invalid values for the required columns.
    """
    df = df.copy()

    # Required columns for analysis
    required_cols = [
        'uuid', 'page_id', 'reader_view', 'speed', 'num_words', 'Flesch_Kincaid',
        'age', 'device', 'dyslexia_bin', 'english_native', 'correct_rate', 'retake_trial'
    ]

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_cols)

    # Ensure numeric columns are numeric
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df['num_words'] = pd.to_numeric(df['num_words'], errors='coerce')
    df['Flesch_Kincaid'] = pd.to_numeric(df['Flesch_Kincaid'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce').astype(int)
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce').astype(int)
    df['correct_rate'] = pd.to_numeric(df['correct_rate'], errors='coerce')
    df['retake_trial'] = pd.to_numeric(df['retake_trial'], errors='coerce').astype(int)

    # Remove rows with non-positive speed (cannot log-transform) or missing after coercion
    df = df[df['speed'] > 0]
    df = df.dropna(subset=['speed', 'num_words', 'Flesch_Kincaid', 'age', 'device', 'english_native', 'correct_rate'])

    # Log-transform dependent variable and text length
    df['log_speed'] = np.log(df['speed'].astype(float))
    # To stabilize the influence of extreme page lengths, use log(num_words)
    df['log_num_words'] = np.log(df['num_words'].astype(float))

    # Derive english native binary: 'Y' -> 1, else 0
    df['english_native_binary'] = df['english_native'].apply(lambda x: 1 if str(x).strip().upper() == 'Y' else 0)

    # Ensure device is categorical
    df['device'] = df['device'].astype('category')

    # Optional: remove extreme outliers in log_speed by winsorizing at 1st and 99th percentiles to reduce undue influence
    # (keeps shape while limiting extreme points). This is conservative and can be commented out if undesired.
    lower = df['log_speed'].quantile(0.01)
    upper = df['log_speed'].quantile(0.99)
    df['log_speed'] = df['log_speed'].clip(lower, upper)

    # Final dropna to be safe
    final_cols = [
        'uuid', 'page_id', 'reader_view', 'log_speed', 'dyslexia_bin', 'log_num_words',
        'Flesch_Kincaid', 'age', 'device', 'english_native_binary', 'correct_rate', 'retake_trial'
    ]
    df = df.dropna(subset=final_cols)

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> any:
    """
    Fit a linear model testing whether Reader View (reader_view) affects reading speed and whether
    this effect is different for readers with dyslexia (interaction reader_view * dyslexia_bin).

    Model: log_speed ~ reader_view * dyslexia_bin + log_num_words + Flesch_Kincaid + age + C(device) + english_native_binary + correct_rate + retake_trial

    Uses cluster-robust standard errors clustered by participant UUID to account for repeated measures.

    Returns the fitted OLS results object.
    """
    import statsmodels.formula.api as smf

    # Ensure required columns exist
    required_model_cols = [
        'log_speed', 'reader_view', 'dyslexia_bin', 'log_num_words', 'Flesch_Kincaid',
        'age', 'device', 'english_native_binary', 'correct_rate', 'retake_trial', 'uuid'
    ]
    missing = [c for c in required_model_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for model: {missing}")

    formula = (
        'log_speed ~ reader_view * dyslexia_bin + log_num_words + Flesch_Kincaid + '
        'age + C(device) + english_native_binary + correct_rate + retake_trial'
    )

    ols_mod = smf.ols(formula=formula, data=df)
    # Cluster robust standard errors clustered on participant UUID
    results = ols_mod.fit(cov_type='cluster', cov_kwds={'groups': df['uuid']})

    # Return the fitted results object (has .summary(), .params, .bse, etc.)
    return results


