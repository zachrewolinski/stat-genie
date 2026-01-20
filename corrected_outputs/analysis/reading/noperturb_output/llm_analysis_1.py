from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and derive variables needed for the analysis.

    Steps performed:
    - Make a copy of the input df.
    - Drop rows with missing values in core columns used to compute wpm or key IV/moderator/cluster id.
    - Exclude trials with non-positive adjusted_running_time or non-positive num_words.
    - Exclude retake trials (retake_trial == 1) when present.
    - Compute wpm = num_words * 60000 / adjusted_running_time.
    - Remove extreme wpm outliers (abs(z) > 4).
    - Ensure variable types for modeling (ints, strings where appropriate).
    - Return dataframe with the exact columns used in the model.
    """
    df = df.copy()

    # Required columns for the computation and clustering
    required_cols = [
        'adjusted_running_time', 'num_words', 'reader_view', 'dyslexia_bin', 'uuid'
    ]

    # Drop rows missing core variables (including uuid for clustering)
    df = df.dropna(subset=required_cols)

    # Ensure numeric conversions where needed
    df['adjusted_running_time'] = pd.to_numeric(df['adjusted_running_time'], errors='coerce')
    df['num_words'] = pd.to_numeric(df['num_words'], errors='coerce')
    df['reader_view'] = pd.to_numeric(df['reader_view'], errors='coerce')
    df['dyslexia_bin'] = pd.to_numeric(df['dyslexia_bin'], errors='coerce')

    # After coercion, drop any rows that became NaN in these required numeric columns
    df = df.dropna(subset=['adjusted_running_time', 'num_words', 'reader_view', 'dyslexia_bin', 'uuid'])

    # Remove rows with non-positive times or word counts
    df = df[df['adjusted_running_time'] > 0]
    df = df[df['num_words'] > 0]

    # Exclude retake trials if the column exists (retakes may not reflect natural reading)
    if 'retake_trial' in df.columns:
        df = df[df['retake_trial'] == 0]

    # Compute words-per-minute from adjusted running time (adjusted_running_time in ms)
    # wpm = num_words / (minutes) = num_words / (ms / 60000) = num_words * 60000 / ms
    df['wpm'] = df['num_words'] * 60000.0 / df['adjusted_running_time']

    # Remove extreme outliers in wpm (e.g., measurement errors). Use z-score threshold of 4.
    # Use population std (ddof=0) to match previous behavior
    wpm_mean = df['wpm'].mean()
    wpm_std = df['wpm'].std(ddof=0)
    if wpm_std > 0:
        df['wpm_z'] = (df['wpm'] - wpm_mean) / wpm_std
        df = df[df['wpm_z'].abs() <= 4]
        df = df.drop(columns=['wpm_z'])

    # Coerce types and prepare categorical vars used in the formula
    # Ensure binary indicators are integers 0/1
    df['dyslexia_bin'] = df['dyslexia_bin'].astype(int)
    df['reader_view'] = df['reader_view'].astype(int)

    # Keep categorical columns as strings (statsmodels formula will handle them with C(...))
    if 'english_native' in df.columns:
        # Map common boolean-like values to 'Y'/'N' if present, otherwise cast to string
        df['english_native'] = df['english_native'].astype(str)

    if 'device' in df.columns:
        df['device'] = df['device'].astype(str)
    if 'page_id' in df.columns:
        df['page_id'] = df['page_id'].astype(str)

    # Select and return only the columns needed for modeling (keep uuid for clustering)
    keep_cols = [
        'uuid', 'page_id', 'reader_view', 'dyslexia_bin', 'wpm', 'age', 'device',
        'Flesch_Kincaid', 'num_words', 'img_width', 'correct_rate', 'english_native'
    ]

    # Some datasets may not contain all optional columns; keep those present
    keep_cols = [c for c in keep_cols if c in df.columns]

    df = df[keep_cols].reset_index(drop=True)

    return df


def model(df: pd.DataFrame) -> Any:
    """
    Fit a linear model testing whether Reader View affects reading speed (wpm) and whether
    this effect is moderated by dyslexia (dyslexia_bin). The model includes relevant controls
    and clusters standard errors by participant (uuid) to account for repeated measures.

    Model formula:
    wpm ~ reader_view * dyslexia_bin + age + C(device) + Flesch_Kincaid + num_words + img_width + correct_rate + C(page_id) + C(english_native)

    Returns the fitted statsmodels regression results object (OLS with cluster-robust SEs by uuid).
    """
    # Ensure the cluster variable is present
    if 'uuid' not in df.columns:
        raise ValueError("The transformed dataframe must include 'uuid' for clustering standard errors.")

    # Build formula: include interaction between reader_view and dyslexia_bin
    formula_terms = [
        'reader_view * dyslexia_bin',
        'age',
        'C(device)',
        'Flesch_Kincaid',
        'num_words',
        'img_width',
        'correct_rate',
        'C(page_id)',
        'C(english_native)'
    ]

    # Keep only terms whose columns exist in df (robust to missing optional controls)
    available_terms = []
    for term in formula_terms:
        # For categorical C(x) terms, check that x exists
        if term.startswith('C(') and term.endswith(')'):
            col = term[2:-1]
            if col in df.columns:
                available_terms.append(term)
        else:
            # For interaction and normal terms, parse base column names and check existence
            base_cols = [c.strip() for c in term.replace('*', '+').split('+')]
            if all((bc in df.columns) for bc in base_cols):
                available_terms.append(term)

    if not available_terms:
        raise ValueError("No predictor terms are available in the dataframe to build the model.")

    formula = 'wpm ~ ' + ' + '.join(available_terms)

    # Build the model (this constructs the design matrix and determines which rows are used)
    model_obj = smf.ols(formula, data=df)

    # Prepare cluster groups aligned to the rows actually used by the model.
    # Patsy may drop rows with missing values in any term used in the formula,
    # so we must select group labels corresponding to the model's row labels.
    # Create a Series indexed by the dataframe's index to allow alignment by label.
    uuid_series = pd.Series(df['uuid'].values, index=df.index)

    # model_obj.data.row_labels contains the index labels (from df) used in the model design matrix
    row_labels = getattr(model_obj.data, "row_labels", None)
    if row_labels is None:
        # Fallback: when unavailable, assume all rows in df were used
        row_labels = df.index.tolist()

    # Align uuid values to the rows used by the model
    groups_series = uuid_series.reindex(row_labels)

    # Convert to integer codes for grouping (statsmodels accepts integer or array-like)
    groups = groups_series.astype('category').cat.codes.values

    # Confirm groups length matches the model's expected observation count
    n_obs_model = len(row_labels)
    if len(groups) != n_obs_model:
        # As a last resort, align by position: take first n_obs_model entries
        groups = np.asarray(groups)[:n_obs_model]

    # Fit OLS and cluster standard errors by uuid (accounts for within-participant correlation)
    results = model_obj.fit(cov_type='cluster', cov_kwds={'groups': groups})

    # Return the fitted results object (has .summary(), params, conf_int(), etc.)
    return results