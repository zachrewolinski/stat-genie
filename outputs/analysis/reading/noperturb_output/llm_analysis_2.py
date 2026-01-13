from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/outputs/analysis/reading/noperturb_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into analysis-ready dataframe.

    - Drops rows with missing critical values.
    - Excludes retake trials (if retake_trial column exists).
    - Filters out non-positive or implausibly small adjusted_running_time.
    - Computes reading_speed_wpm from num_words and adjusted_running_time (ms -> minutes).
    - Computes log_reading_speed = log(reading_speed_wpm).
    - Encodes english_native into binary english_native_bin (Y->1, N->0).

    Returns a dataframe containing at least the columns used in the model:
      ['uuid','page_id','reader_view','dyslexia_bin','reading_speed_wpm','log_reading_speed',
       'age','num_words','Flesch_Kincaid','correct_rate','english_native_bin','device']
    """
    df = df.copy()

    # Ensure required columns exist; if not, raise informative error
    required_cols = ['adjusted_running_time', 'num_words', 'reader_view', 'dyslexia_bin', 'uuid', 'page_id']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")

    # Convert core numeric columns safely
    df['adjusted_running_time'] = pd.to_numeric(df['adjusted_running_time'], errors='coerce')
    df['num_words'] = pd.to_numeric(df['num_words'], errors='coerce')

    # Convert potential numeric covariates if present
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    if 'Flesch_Kincaid' in df.columns:
        df['Flesch_Kincaid'] = pd.to_numeric(df['Flesch_Kincaid'], errors='coerce')
    if 'correct_rate' in df.columns:
        df['correct_rate'] = pd.to_numeric(df['correct_rate'], errors='coerce')

    # Convert reader_view and dyslexia_bin to numeric (regular numpy dtypes, not pandas' nullable Int64)
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce')
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce')

    # Remove retake trials if present (retake_trial == 1 indicates a retake)
    if 'retake_trial' in df.columns:
        df = df[df['retake_trial'] == 0]

    # Filter out rows with missing or non-positive adjusted_running_time or num_words or key indicators
    df = df.dropna(subset=['adjusted_running_time', 'num_words', 'reader_view', 'dyslexia_bin', 'uuid', 'page_id'])
    df = df[df['adjusted_running_time'] > 200]  # remove implausibly short durations (ms)
    df = df[df['num_words'] > 0]

    # Now safe to cast reader_view and dyslexia_bin to integer numpy dtypes
    df['reader_view'] = df['reader_view'].astype(int)
    df['dyslexia_bin'] = df['dyslexia_bin'].astype(int)

    # Compute reading speed in words per minute (wpm). adjusted_running_time is in milliseconds.
    df['reading_speed_wpm'] = df['num_words'] / (df['adjusted_running_time'] / 60000.0)

    # Keep only positive speeds
    df = df[df['reading_speed_wpm'] > 0]

    # Log-transform the reading speed for modeling
    df['log_reading_speed'] = np.log(df['reading_speed_wpm'])

    # Encode english_native into binary indicator english_native_bin (Y -> 1, else 0)
    if 'english_native' in df.columns:
        df['english_native_bin'] = df['english_native'].map({'Y': 1, 'N': 0}).fillna(0).astype(int)
    else:
        # Ensure column exists in final dataframe per contract
        df['english_native_bin'] = 0

    # Ensure Flesch_Kincaid and correct_rate numeric (already attempted above)
    if 'Flesch_Kincaid' in df.columns:
        df['Flesch_Kincaid'] = pd.to_numeric(df['Flesch_Kincaid'], errors='coerce')
    if 'correct_rate' in df.columns:
        df['correct_rate'] = pd.to_numeric(df['correct_rate'], errors='coerce')

    # Ensure all desired final columns exist (add as NaN/default if missing)
    desired = ['uuid', 'page_id', 'reader_view', 'dyslexia_bin', 'reading_speed_wpm', 'log_reading_speed',
               'age', 'num_words', 'Flesch_Kincaid', 'correct_rate', 'english_native_bin', 'device']
    for col in desired:
        if col not in df.columns:
            # Add missing columns with appropriate default NA values.
            # Keep types flexible; modeling will handle missingness by dropping rows if necessary.
            df[col] = np.nan

    # Ensure device stays as object/string if present; leave NaN otherwise
    if 'device' in df.columns:
        df['device'] = df['device'].astype(object)

    # Reset index and return only the desired columns in the specified order
    df = df[desired].reset_index(drop=True)

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    """
    Fit an OLS model testing whether Reader View improves reading speed for readers with dyslexia.

    Model formula:
      log_reading_speed ~ reader_view * dyslexia_bin + age + num_words + Flesch_Kincaid + correct_rate + english_native_bin + C(device) + C(page_id)

    - Interaction reader_view * dyslexia_bin tests whether the Reader View effect differs for dyslexic readers.
    - C(device) and C(page_id) add categorical controls (dummy variables) for device and page.
    - Standard errors are clustered by participant UUID to account for within-subject dependence.

    Returns the fitted model results with cluster-robust standard errors.
    """
    # Check the required dependent variable is present
    if 'log_reading_speed' not in df.columns:
        raise ValueError("Transformed dataframe must contain 'log_reading_speed' for modeling. Run transform() first.")

    # Ensure numeric covariates are numeric types (coerce if necessary)
    df = df.copy()
    df['num_words'] = pd.to_numeric(df['num_words'], errors='coerce')
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    if 'Flesch_Kincaid' in df.columns:
        df['Flesch_Kincaid'] = pd.to_numeric(df['Flesch_Kincaid'], errors='coerce')
    if 'correct_rate' in df.columns:
        df['correct_rate'] = pd.to_numeric(df['correct_rate'], errors='coerce')
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce').astype(float)
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce').astype(float)
    df['english_native_bin'] = pd.to_numeric(df['english_native_bin'], errors='coerce').astype(float)

    # Build the formula. Use categorical controls for device and page_id if present.
    formula_terms = ['reader_view * dyslexia_bin', 'age', 'num_words', 'Flesch_Kincaid', 'correct_rate', 'english_native_bin']
    if 'device' in df.columns:
        formula_terms.append('C(device)')
    if 'page_id' in df.columns:
        formula_terms.append('C(page_id)')

    formula = 'log_reading_speed ~ ' + ' + '.join(formula_terms)

    # Fit OLS; statsmodels/patsy will drop rows with missing data automatically
    mod = smf.ols(formula=formula, data=df)
    fit = mod.fit()

    # Cluster-robust SEs by participant UUID if uuid column available.
    # Need to align group labels with the rows actually used in the fitted model.
    if 'uuid' in df.columns:
        try:
            # fit.model.data.row_labels are the original index labels for the rows used in the fit.
            used_index = fit.model.data.row_labels
            # Select group labels in the same order. Convert to integer codes required by some statsmodels implementations.
            groups_series = df.loc[used_index, 'uuid']
            groups = pd.Categorical(groups_series).codes
            results = fit.get_robustcov_results(cov_type='cluster', groups=groups)
        except Exception:
            # As a fallback, try to compute groups from the fitted design's index in a safe way.
            try:
                used_index = fit.model.data.row_labels
                groups_series = df.loc[used_index, 'uuid']
                # If there is any problem converting to categorical codes above, try a numpy array of values.
                groups = np.asarray(groups_series)
                # Some statsmodels versions require integer group codes; attempt categorical codes if possible.
                try:
                    groups = pd.Categorical(groups_series).codes
                except Exception:
                    groups = np.asarray(groups_series)
                results = fit.get_robustcov_results(cov_type='cluster', groups=groups)
            except Exception:
                # If everything fails, return the original fit (no clustering).
                results = fit
    else:
        results = fit

    return results