from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/reading/replace_with_rvs_output/reading.csv')

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe.

    Returns a dataframe with the exact columns required by the model:
    ['uuid', 'page_id', 'reader_view', 'dyslexia_bin', 'age', 'Flesch_Kincaid',
     'num_words', 'img_width', 'english_native_bin', 'log_wps', 'device']
    """
    df = df.copy()

    # Drop rows missing essential columns for computing the DV and primary IVs.
    # Note: page_id/device may be missing in raw data; we'll add placeholders later.
    required_cols = [
        'adjusted_running_time',
        'num_words',
        'reader_view',
        'dyslexia_bin',
        'uuid',
        'correct_rate',
    ]
    # Only drop based on the subset of required_cols that actually exist in the raw df
    existing_required = [c for c in required_cols if c in df.columns]
    if existing_required:
        df = df.dropna(subset=existing_required)

    # Exclude retake trials (keep first attempts only)
    if 'retake_trial' in df.columns:
        df = df[df['retake_trial'] == 0]

    # Keep only positive adjusted running time and positive word counts (if present)
    if 'adjusted_running_time' in df.columns:
        df = df[df['adjusted_running_time'] > 0]
    if 'num_words' in df.columns:
        df = df[df['num_words'] > 0]

    # Require a minimum comprehension threshold (adjustable)
    if 'correct_rate' in df.columns:
        df = df[df['correct_rate'] >= 0.5]

    # Compute words per second using adjusted_running_time (milliseconds -> seconds)
    if ('adjusted_running_time' in df.columns) and ('num_words' in df.columns):
        # Guard against division by zero already handled above by > 0 checks
        df['words_per_sec'] = df['num_words'] / (df['adjusted_running_time'] / 1000.0)
    else:
        # Could not compute words_per_sec from raw data; fill with NA so downstream code can handle it
        df['words_per_sec'] = pd.NA

    # Drop infinite / NaN and keep only positive words_per_sec
    df['words_per_sec'] = df['words_per_sec'].replace([np.inf, -np.inf], pd.NA)
    df = df.dropna(subset=['words_per_sec'])
    df = df[df['words_per_sec'] > 0]

    # Dependent variable: log-transformed words-per-second
    # If words_per_sec contains valid values, compute log; otherwise will be handled by dropna above.
    df['log_wps'] = np.log(df['words_per_sec'])

    # Encode english native speaker indicator
    if 'english_native' in df.columns:
        df['english_native_bin'] = df['english_native'].map({'Y': 1, 'N': 0})
        df['english_native_bin'] = df['english_native_bin'].fillna(0).astype(int)
    else:
        df['english_native_bin'] = 0

    # Winsorize/clamp log_wps to reduce influence of extreme outliers (1st-99th percentile)
    if 'log_wps' in df.columns and not df['log_wps'].empty:
        lower = df['log_wps'].quantile(0.01)
        upper = df['log_wps'].quantile(0.99)
        # Only clip if quantiles are finite numbers
        if np.isfinite(lower) and np.isfinite(upper):
            df['log_wps'] = df['log_wps'].clip(lower=lower, upper=upper)

    # Keep only the columns needed for the analysis and modeling
    out_cols = [
        'uuid',
        'page_id',
        'reader_view',
        'dyslexia_bin',
        'age',
        'Flesch_Kincaid',
        'num_words',
        'img_width',
        'english_native_bin',
        'log_wps',
        'device',
    ]

    # Add missing optional columns with placeholders or NA as appropriate
    for c in out_cols:
        if c not in df.columns:
            if c in ['device', 'page_id', 'uuid']:
                df[c] = 'missing'
            else:
                df[c] = pd.NA

    # Ensure numeric columns are numeric where appropriate
    numeric_cols = ['age', 'Flesch_Kincaid', 'num_words', 'img_width']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Ensure key binary columns are integer typed where possible (coerce non-numeric -> 0)
    # Use fillna(0) so that missing values become 0 (no Reader View, no dyslexia) in the final DF
    if 'reader_view' in df.columns:
        df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce').fillna(0).astype(int)
    else:
        df['reader_view'] = 0

    if 'dyslexia_bin' in df.columns:
        df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce').fillna(0).astype(int)
    else:
        df['dyslexia_bin'] = 0

    # Subset to the final columns (this will keep the existing row index)
    df = df[out_cols]

    # Convert categorical identifier columns to categorical dtype with at least one category.
    # Ensure there's always at least the 'missing' category to avoid zero-length categories (which breaks patsy).
    for cat_col in ['device', 'page_id', 'uuid']:
        # Convert to strings, fill missing marker
        s = df[cat_col].fillna('missing').astype(str)
        # Determine categories from present values; always include 'missing'
        unique_vals = list(pd.unique(s))
        if 'missing' not in unique_vals:
            unique_vals.append('missing')
        if len(unique_vals) == 0:
            unique_vals = ['missing']
        df[cat_col] = pd.Categorical(s, categories=unique_vals)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model testing whether Reader View improves reading speed for readers with dyslexia.

    Model:
    log_wps ~ reader_view * dyslexia_bin + age + Flesch_Kincaid + num_words + img_width + english_native_bin + C(device) + C(page_id)

    Clustered standard errors by participant uuid where possible.
    """
    # Drop rows missing essential modeling columns
    model_df = df.dropna(subset=['log_wps', 'reader_view', 'dyslexia_bin', 'uuid'])

    # If the model dataframe is empty, return None to indicate no model could be fit
    if model_df.shape[0] == 0:
        return None

    # Ensure numeric predictors are numeric (coerce invalid values to NaN, then drop if needed)
    numeric_preds = ['log_wps', 'reader_view', 'dyslexia_bin', 'age', 'Flesch_Kincaid', 'num_words', 'img_width', 'english_native_bin']
    for col in numeric_preds:
        if col in model_df.columns:
            model_df[col] = pd.to_numeric(model_df[col], errors='coerce')

    # After coercion, ensure still have required non-NA rows
    model_df = model_df.dropna(subset=['log_wps', 'reader_view', 'dyslexia_bin', 'uuid'])
    if model_df.shape[0] == 0:
        return None

    # Construct formula including categorical fixed effects for device and page_id
    formula = (
        'log_wps ~ reader_view * dyslexia_bin + age + Flesch_Kincaid + num_words + img_width + '
        'english_native_bin + C(device) + C(page_id)'
    )

    # Fit OLS safely
    try:
        ols_fit = smf.ols(formula, data=model_df).fit()
    except Exception:
        # If fitting fails for any reason, return None rather than raising an unhandled exception
        return None

    # Attempt clustered robust covariance by participant (uuid). If it fails, return the plain ols_fit
    try:
        clustered = ols_fit.get_robustcov_results(cov_type='cluster', groups=model_df['uuid'])
        results = clustered
    except Exception:
        try:
            results = ols_fit.get_robustcov_results(cov_type='HC3')
        except Exception:
            results = ols_fit

    return results