from typing import Any
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('/accounts/grad/zachrewolinski/research/stat-genie/corrected_outputs/analysis/reading/replace_with_rvs_output/reading.csv')

# ======== TRANSFORM CODE ========
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Keep only rows with required columns present
    required_cols = ['speed', 'reader_view', 'dyslexia_bin', 'uuid']
    df = df.dropna(subset=required_cols)

    # Remove non-positive speeds (cannot log-transform)
    df = df[df['speed'] > 0]

    # Dependent variable: log-transform speed to reduce skew
    df['log_speed'] = np.log(df['speed'])

    # Independent and moderator: ensure binary/int types
    df['reader_view'] = df['reader_view'].astype(int)
    df['dyslexia_bin'] = df['dyslexia_bin'].astype(int)

    # English native map to binary (Y -> 1, N -> 0). If missing, set to 0 (non-native) conservatively.
    if 'english_native' in df.columns:
        df['english_native_bin'] = df['english_native'].map({'Y': 1, 'N': 0})
        df['english_native_bin'] = df['english_native_bin'].fillna(0).astype(int)
    else:
        df['english_native_bin'] = 0

    # Mean-center continuous controls to help interpretation
    for col in ['age', 'num_words', 'Flesch_Kincaid']:
        if col in df.columns:
            mean_val = df[col].mean()
            df[f'{col}_c'] = df[col] - mean_val
        else:
            # create a zero column if missing to keep model code stable
            df[f'{col}_c'] = 0.0

    # Ensure other control columns exist and have appropriate dtypes
    if 'retake_trial' not in df.columns:
        df['retake_trial'] = 0
    else:
        df['retake_trial'] = df['retake_trial'].fillna(0).astype(int)

    if 'correct_rate' not in df.columns:
        df['correct_rate'] = 0.0
    else:
        df['correct_rate'] = df['correct_rate'].fillna(0.0)

    # Device, gender, page_id: convert to categorical for use with C(...) in statsmodels
    for catcol in ['device', 'gender', 'page_id']:
        if catcol in df.columns:
            df[catcol] = df[catcol].astype('category')
        else:
            # create a default category if missing
            df[catcol] = 'missing'
            df[catcol] = df[catcol].astype('category')

    # Keep only columns needed for modeling (plus original keys if desired)
    keep_cols = [
        'log_speed', 'reader_view', 'dyslexia_bin', 'age_c', 'num_words_c', 'Flesch_Kincaid_c',
        'retake_trial', 'correct_rate', 'english_native_bin', 'device', 'gender', 'page_id', 'uuid'
    ]

    # If any of the keep_cols are missing (shouldn't be), fill with defaults
    for k in keep_cols:
        if k not in df.columns:
            if k in ['reader_view', 'dyslexia_bin', 'retake_trial', 'english_native_bin']:
                df[k] = 0
            elif k in ['log_speed', 'age_c', 'num_words_c', 'Flesch_Kincaid_c', 'correct_rate']:
                df[k] = 0.0
            else:
                df[k] = 'missing'

    df = df[keep_cols]

    return df


# ======== MODEL CODE ========
def model(df: pd.DataFrame) -> Any:
    # Formula includes interaction between reader_view and dyslexia_bin (test for differential effect)
    # Controls: mean-centered continuous covariates and categorical controls via C(...)
    formula = (
        'log_speed ~ reader_view * dyslexia_bin + age_c + num_words_c + Flesch_Kincaid_c '
        '+ retake_trial + correct_rate + english_native_bin + C(device) + C(gender) + C(page_id)'
    )

    # Fit OLS
    ols_model = smf.ols(formula, data=df).fit()

    # Obtain cluster-robust standard errors clustered by participant uuid (accounts for repeated measures)
    try:
        results = ols_model.get_robustcov_results(cov_type='cluster', groups=df['uuid'])
    except Exception:
        # Fall back to heteroskedasticity-robust (HC3) if clustering fails
        results = ols_model.get_robustcov_results(cov_type='HC3')

    # Print a concise, robust summary table (avoid calling results.summary() which can fail with some robust cov types)
    try:
        coef = results.params
        se = results.bse
        tvals = results.tvalues
        pvals = results.pvalues
        try:
            conf = results.conf_int()
            conf.columns = ['ci_lower', 'ci_upper']
            summary_df = pd.concat([coef.rename('coef'), se.rename('std_err'), tvals.rename('t'), pvals.rename('pval'), conf], axis=1)
        except Exception:
            summary_df = pd.concat([coef.rename('coef'), se.rename('std_err'), tvals.rename('t'), pvals.rename('pval')], axis=1)

        print(summary_df.to_string(float_format=lambda x: f'{x:.4f}'))
    except Exception:
        # As a last resort, print parameter values only
        print("Could not produce full summary; parameters:")
        print(results.params)

    return results