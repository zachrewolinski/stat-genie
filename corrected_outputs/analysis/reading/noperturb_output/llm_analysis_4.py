from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/reading/noperturb_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataframe for modeling. Steps:
    - Drop rows missing key variables (speed, reader_view, dyslexia_bin).
    - Create log_speed = ln(speed). Add small epsilon to avoid log(0) if necessary.
    - Ensure binary / numeric columns are numeric and in expected ranges.
    - Create english_native_Y binary indicator from english_native ('Y'/'N').
    - Winsorize log_speed at 1st and 99th percentiles to reduce influence of extreme outliers.
    - Convert device and page_id to categorical dtype (kept as columns for formula-based factor controls).
    - Return dataframe containing all columns needed by the model.
    """

    # Work on a copy
    df = df.copy()

    # Required columns for modeling
    required_cols = [
        'speed', 'reader_view', 'dyslexia_bin', 'num_words', 'Flesch_Kincaid',
        'age', 'english_native', 'retake_trial', 'device', 'page_id', 'correct_rate'
    ]

    # Drop rows with missing values in required columns
    df = df.dropna(subset=['speed', 'reader_view', 'dyslexia_bin'])

    # Coerce numeric columns to numeric types where appropriate
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce').fillna(0).astype(int)
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce')
    df['num_words'] = pd.to_numeric(df['num_words'], errors='coerce')
    df['Flesch_Kincaid'] = pd.to_numeric(df['Flesch_Kincaid'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['retake_trial'] = pd.to_numeric(df['retake_trial'], errors='coerce')
    df['correct_rate'] = pd.to_numeric(df['correct_rate'], errors='coerce')

    # Re-drop rows that became NaN after coercion in essential columns
    df = df.dropna(subset=['speed', 'reader_view', 'dyslexia_bin'])

    # Create log_speed (add small epsilon to avoid -inf)
    eps = 1e-6
    df['log_speed'] = np.log(df['speed'] + eps)

    # Winsorize log_speed at 1st and 99th percentiles to limit extreme influence
    lower = df['log_speed'].quantile(0.01)
    upper = df['log_speed'].quantile(0.99)
    df['log_speed'] = df['log_speed'].clip(lower=lower, upper=upper)

    # Ensure dyslexia_bin is binary 0/1 (if original had 2 for severe dyslexia, we keep it as 1 in dyslexia_bin if column already encodes binary;
    # dataset includes dyslexia_bin already which is 1 = dyslexia, 0 = no dyslexia. If not, map non-zero -> 1)
    df['dyslexia_bin'] = df['dyslexia_bin'].apply(lambda x: 1 if pd.notnull(x) and x != 0 else 0).astype(int)

    # Create english_native_Y indicator (1 if 'Y', 0 otherwise). If english_native is missing, set to 0.
    df['english_native_Y'] = df['english_native'].apply(lambda x: 1 if str(x).strip().upper() == 'Y' else 0)

    # Convert device and page_id to categorical dtype (kept as-is for usage with formula 'C(device)' etc.)
    if 'device' in df.columns:
        df['device'] = df['device'].astype('category')
    if 'page_id' in df.columns:
        df['page_id'] = df['page_id'].astype('category')

    # For safety, ensure retake_trial is binary 0/1
    df['retake_trial'] = df['retake_trial'].fillna(0).apply(lambda x: 1 if x == 1 else 0).astype(int)

    # Keep only rows with plausible ages (>= 8 and <= 100) to remove invalid ages if present
    df.loc[~df['age'].between(8, 100), 'age'] = np.nan

    # Drop rows that are now missing critical controls (if desired). We will keep rows missing some non-critical controls but drop when modeling will error.
    # Here drop rows missing the main controls that we plan to include in the model
    model_cols = ['log_speed', 'reader_view', 'dyslexia_bin', 'num_words', 'Flesch_Kincaid', 'age', 'english_native_Y', 'retake_trial', 'device', 'page_id', 'correct_rate']
    df = df.dropna(subset=['log_speed', 'reader_view', 'dyslexia_bin'])

    # Return the dataframe with all columns required for modeling. Extra columns from original DF are preserved.
    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame):
    """
    Fit an OLS model to estimate the effect of Reader View on reading speed and whether that effect differs for readers with dyslexia.

    Model specification (main):
    log_speed ~ reader_view * dyslexia_bin + num_words + Flesch_Kincaid + age + english_native_Y + retake_trial + C(device) + C(page_id) + correct_rate

    - Interaction reader_view * dyslexia_bin tests whether the Reader View effect differs for dyslexic readers.
    - C(device) and C(page_id) include categorical fixed effects for device and page.
    - Use robust (HC3) standard errors to protect against heteroskedasticity.

    Returns the fitted results object from statsmodels (RegressionResultsWrapper).
    """

    import statsmodels.formula.api as smf

    formula = (
        'log_speed ~ reader_view * dyslexia_bin + num_words + Flesch_Kincaid + age '
        '+ english_native_Y + retake_trial + correct_rate + C(device) + C(page_id)'
    )

    # Fit OLS with heteroskedasticity-robust (HC3) standard errors
    model = smf.ols(formula, data=df).fit(cov_type='HC3')

    # It's helpful to inspect the summary in interactive use; we return the results object so callers can print/inspect.
    # Example: print(model.summary())
    return model


