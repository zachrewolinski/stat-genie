from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the raw dataframe into the analysis-ready dataframe.

    Output columns required by the model:
      - reader_view: int (0/1)
      - dyslexia_bin: int (0/1)
      - log_speed: float (natural log of speed + 1)
      - num_words: numeric
      - Flesch_Kincaid: numeric
      - age_c: numeric (age centered)
      - retake_trial: int (0/1)
      - page_id: categorical
      - device: categorical
      - english_native: categorical
      - uuid: identifier (used to cluster SEs)
    """
    df = df.copy()

    # Required columns for analysis: list (we'll check existence below)
    required = [
        'speed',
        'reader_view',
        'dyslexia',           # optional if dyslexia_bin present
        'dyslexia_bin',       # optional if dyslexia present
        'num_words',
        'Flesch_Kincaid',
        'age',
        'retake_trial',
        'page_id',
        'device',
        'english_native',
        'uuid',
    ]

    # Coerce numeric columns that we'll immediately operate on
    if 'speed' in df.columns:
        df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    else:
        raise ValueError('Required column for analysis missing: speed')

    # Ensure reader_view is numeric-ish before dropping rows
    if 'reader_view' in df.columns:
        df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce')
    else:
        raise ValueError('Required column for analysis missing: reader_view')

    # Drop rows missing essential variables speed or reader_view
    df = df.dropna(subset=['speed', 'reader_view'])

    # Handle dyslexia_bin/dyslexia robustly without converting to int until after NA rows are removed
    if 'dyslexia_bin' in df.columns:
        # Coerce to numeric; keep NaN for rows where it's missing/invalid
        df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce')
    elif 'dyslexia' in df.columns:
        # Coerce dyslexia to numeric then derive dyslexia_bin where dyslexia is present
        df['dyslexia'] = pd.to_numeric(df['dyslexia'], errors='coerce')
        df['dyslexia_bin'] = df['dyslexia'].apply(
            lambda x: 1 if (pd.notnull(x) and x > 0) else (0 if pd.notnull(x) and x == 0 else np.nan)
        )
    else:
        raise ValueError('Neither dyslexia_bin nor dyslexia column present in dataframe')

    # Create the log-transformed dependent variable to reduce skew
    df['log_speed'] = np.log(df['speed'].astype(float) + 1.0)

    # Center age for interpretability (use available ages only)
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        age_mean = df['age'].mean(skipna=True)
        df['age_c'] = df['age'] - age_mean
    else:
        df['age_c'] = np.nan

    # Ensure numeric controls exist and raise if missing required columns
    for col in ['num_words', 'Flesch_Kincaid', 'retake_trial', 'uuid', 'page_id', 'device', 'english_native']:
        if col not in df.columns:
            raise ValueError(f"Required column for analysis missing: {col}")

    # Coerce num_words and Flesch_Kincaid to numeric
    df['num_words'] = pd.to_numeric(df['num_words'], errors='coerce')
    df['Flesch_Kincaid'] = pd.to_numeric(df['Flesch_Kincaid'], errors='coerce')

    # Cast categorical variables to categories (keeps them for formula-based modeling)
    df['page_id'] = df['page_id'].astype('category')
    df['device'] = df['device'].astype('category')
    df['english_native'] = df['english_native'].astype('category')

    # Ensure retake_trial numeric and fill missing with 0 (assume missing => not a retake)
    df['retake_trial'] = pd.to_numeric(df['retake_trial'], errors='coerce').fillna(0)

    # Now drop rows with any missing values in model columns (this will remove rows where dyslexia_bin is NaN)
    model_cols = [
        'log_speed',
        'reader_view',
        'dyslexia_bin',
        'num_words',
        'Flesch_Kincaid',
        'age_c',
        'retake_trial',
        'page_id',
        'device',
        'english_native',
        'uuid',
    ]
    df = df.dropna(subset=model_cols)

    # After removing rows with NaNs, it's safe to cast to integer types for binary indicators
    df['reader_view'] = df['reader_view'].astype(int)
    df['dyslexia_bin'] = df['dyslexia_bin'].astype(int)
    df['retake_trial'] = df['retake_trial'].astype(int)

    # Reset index
    df = df.reset_index(drop=True)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Runs an OLS regression of log_speed on reader_view, dyslexia_bin, their interaction,
    and controls. Returns the fitted results object with cluster-robust standard errors by uuid.

    Model formula:
      log_speed ~ reader_view * dyslexia_bin + num_words + Flesch_Kincaid + age_c + retake_trial + C(page_id) + C(device) + C(english_native)

    We cluster standard errors by uuid to account for repeated measurements by the same reader.
    """
    # Specify formula with interaction. C(...) tells statsmodels to treat as categorical.
    formula = (
        'log_speed ~ reader_view * dyslexia_bin'
        ' + num_words + Flesch_Kincaid + age_c + retake_trial'
        ' + C(page_id) + C(device) + C(english_native)'
    )

    # Fit OLS
    ols_model = smf.ols(formula, data=df).fit()

    # Cluster-robust covariance by uuid (accounts for within-user correlation across trials)
    try:
        results = ols_model.get_robustcov_results(cov_type='cluster', groups=df['uuid'])
    except Exception:
        # Fallback: use HC3 robust SE if clustering fails
        results = ols_model.get_robustcov_results(cov_type='HC3')

    # Print summary for quick inspection (can be removed if used programmatically)
    print(results.summary())

    return results