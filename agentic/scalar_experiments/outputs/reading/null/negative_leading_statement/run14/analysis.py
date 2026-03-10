import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('reading.csv')

# Filter dyslexic participants (binary indicator)
df = _df[_df['dyslexia_bin'] == 1].copy()

# Basic cleaning
for col in ['speed', 'reader_view', 'adjusted_running_time', 'num_words']:
    if col in df.columns:
        df = df[pd.to_numeric(df[col], errors='coerce').notna()]

# Ensure numeric types
for col in ['speed', 'reader_view', 'adjusted_running_time', 'num_words', 'Flesch_Kincaid', 'img_width', 'age', 'retake_trial']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Group stats
rv1 = df[df['reader_view'] == 1]['speed'].dropna()
rv0 = df[df['reader_view'] == 0]['speed'].dropna()

summary = df.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std'])

# Welch t-test
if len(rv1) > 1 and len(rv0) > 1:
    ttest = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
else:
    ttest = None

# Cohen's d

def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1 = np.var(a, ddof=1)
    s2 = np.var(b, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if sp == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / sp

d_val = cohens_d(rv1, rv0)

# Alternative speed from adjusted running time (words per minute)
if 'adjusted_running_time' in df.columns and 'num_words' in df.columns:
    df['speed_wpm_calc'] = df['num_words'] / df['adjusted_running_time'] * 60000.0
else:
    df['speed_wpm_calc'] = np.nan

rv1_calc = df[df['reader_view'] == 1]['speed_wpm_calc'].dropna()
rv0_calc = df[df['reader_view'] == 0]['speed_wpm_calc'].dropna()

if len(rv1_calc) > 1 and len(rv0_calc) > 1:
    ttest_calc = stats.ttest_ind(rv1_calc, rv0_calc, equal_var=False, nan_policy='omit')
    d_val_calc = cohens_d(rv1_calc, rv0_calc)
else:
    ttest_calc = None
    d_val_calc = np.nan

# Regression on log speed to mitigate skew
# Filter out non-positive speeds
reg_df = df.copy()
reg_df = reg_df[reg_df['speed'] > 0].copy()
reg_df['log_speed'] = np.log(reg_df['speed'])

# Build formula with available categorical columns
cat_cols = []
for c in ['page_id', 'device', 'education', 'gender', 'language', 'english_native']:
    if c in reg_df.columns:
        cat_cols.append(c)

# Keep only columns with more than one unique value to avoid singularities
cat_terms = []
for c in cat_cols:
    if reg_df[c].nunique(dropna=True) > 1:
        cat_terms.append(f"C({c})")

covariates = [
    'num_words',
    'Flesch_Kincaid',
    'img_width',
    'age',
    'retake_trial',
]

covariates = [c for c in covariates if c in reg_df.columns]

formula_parts = ['reader_view'] + cat_terms + covariates
formula = 'log_speed ~ ' + ' + '.join(formula_parts)

try:
    model = smf.ols(formula, data=reg_df).fit(cov_type='HC3')
    coef = float(model.params.get('reader_view', np.nan))
    pval = float(model.pvalues.get('reader_view', np.nan))
    ci = model.conf_int().loc['reader_view'].tolist() if 'reader_view' in model.params.index else [np.nan, np.nan]
except Exception:
    model = None
    coef = np.nan
    pval = np.nan
    ci = [np.nan, np.nan]

# Also a simpler model controlling only for page_id + num_words
simple_terms = ['reader_view']
if 'page_id' in reg_df.columns and reg_df['page_id'].nunique(dropna=True) > 1:
    simple_terms.append('C(page_id)')
if 'num_words' in reg_df.columns:
    simple_terms.append('num_words')

simple_formula = 'log_speed ~ ' + ' + '.join(simple_terms)

try:
    simple_model = smf.ols(simple_formula, data=reg_df).fit(cov_type='HC3')
    simple_coef = float(simple_model.params.get('reader_view', np.nan))
    simple_pval = float(simple_model.pvalues.get('reader_view', np.nan))
except Exception:
    simple_model = None
    simple_coef = np.nan
    simple_pval = np.nan

# Save key results to json for later use
results = {
    'n_total_dyslexic': int(df.shape[0]),
    'n_reader_view_1': int(rv1.shape[0]),
    'n_reader_view_0': int(rv0.shape[0]),
    'summary_speed': summary.to_dict(),
    'ttest_speed': {
        'statistic': float(ttest.statistic) if ttest else None,
        'pvalue': float(ttest.pvalue) if ttest else None,
    },
    'cohens_d_speed': float(d_val) if not np.isnan(d_val) else None,
    'ttest_speed_calc': {
        'statistic': float(ttest_calc.statistic) if ttest_calc else None,
        'pvalue': float(ttest_calc.pvalue) if ttest_calc else None,
    },
    'cohens_d_speed_calc': float(d_val_calc) if not np.isnan(d_val_calc) else None,
    'regression_log_speed': {
        'coef_reader_view': float(coef) if not np.isnan(coef) else None,
        'pvalue_reader_view': float(pval) if not np.isnan(pval) else None,
        'ci_reader_view': [float(ci[0]), float(ci[1])] if all(np.isfinite(ci)) else [None, None],
        'n_obs': int(reg_df.shape[0]),
        'formula': formula,
    },
    'regression_log_speed_simple': {
        'coef_reader_view': float(simple_coef) if not np.isnan(simple_coef) else None,
        'pvalue_reader_view': float(simple_pval) if not np.isnan(simple_pval) else None,
        'n_obs': int(reg_df.shape[0]),
        'formula': simple_formula,
    }
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
