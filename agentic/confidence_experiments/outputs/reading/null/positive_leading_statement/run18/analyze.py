import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('reading.csv')

# Define dyslexia group using dyslexia_bin if present; fallback to dyslexia > 0
if 'dyslexia_bin' in df.columns:
    dys = df[df['dyslexia_bin'] == 1].copy()
else:
    dys = df[df['dyslexia'] > 0].copy()

# Basic cleanup: drop missing speed or reader_view
cols_needed = ['speed', 'reader_view']
dys = dys.dropna(subset=cols_needed)

# Split groups
rv1 = dys[dys['reader_view'] == 1]['speed']
rv0 = dys[dys['reader_view'] == 0]['speed']

# Descriptive stats
summary = {
    'n_total': int(len(dys)),
    'n_rv1': int(rv1.shape[0]),
    'n_rv0': int(rv0.shape[0]),
    'mean_rv1': float(rv1.mean()) if len(rv1) else np.nan,
    'mean_rv0': float(rv0.mean()) if len(rv0) else np.nan,
    'median_rv1': float(rv1.median()) if len(rv1) else np.nan,
    'median_rv0': float(rv0.median()) if len(rv0) else np.nan,
}

# Welch t-test on raw speed
if len(rv1) > 1 and len(rv0) > 1:
    t_res = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
else:
    t_res = None

# Mann-Whitney U test (non-parametric)
if len(rv1) > 0 and len(rv0) > 0:
    try:
        u_res = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    except ValueError:
        u_res = None
else:
    u_res = None

# Effect size (Cohen's d)
if len(rv1) > 1 and len(rv0) > 1:
    m1, m0 = rv1.mean(), rv0.mean()
    s1, s0 = rv1.std(ddof=1), rv0.std(ddof=1)
    n1, n0 = len(rv1), len(rv0)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
    cohen_d = (m1 - m0) / sp if sp > 0 else np.nan
else:
    cohen_d = np.nan

# Regression controlling for page_id and num_words if available
reg_results = None
if len(dys) > 10:
    # Use log(speed) to reduce skew, add +1 to avoid log(0)
    dys = dys.copy()
    dys['log_speed'] = np.log(dys['speed'] + 1)
    formula_parts = ['reader_view']
    if 'num_words' in dys.columns:
        formula_parts.append('num_words')
    if 'page_id' in dys.columns:
        formula_parts.append('C(page_id)')
    if 'device' in dys.columns:
        formula_parts.append('C(device)')
    if 'age' in dys.columns:
        formula_parts.append('age')
    if 'education' in dys.columns:
        formula_parts.append('C(education)')
    # Build formula
    formula = 'log_speed ~ ' + ' + '.join(formula_parts)
    try:
        model = smf.ols(formula=formula, data=dys).fit(cov_type='HC3')
        reg_results = {
            'coef_reader_view': float(model.params.get('reader_view', np.nan)),
            'p_reader_view': float(model.pvalues.get('reader_view', np.nan)),
            'n_reg': int(model.nobs),
            'r2': float(model.rsquared),
        }
    except Exception:
        reg_results = None

results = {
    'summary': summary,
    't_test': None if t_res is None else {
        'stat': float(t_res.statistic),
        'pvalue': float(t_res.pvalue),
    },
    'mannwhitney': None if u_res is None else {
        'stat': float(u_res.statistic),
        'pvalue': float(u_res.pvalue),
    },
    'cohen_d': float(cohen_d) if np.isfinite(cohen_d) else None,
    'regression': reg_results,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
