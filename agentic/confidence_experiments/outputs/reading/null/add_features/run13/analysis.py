import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('reading.csv')

# Focus on dyslexia participants (binary indicator) and reader_view
# Use dyslexia_bin == 1; drop missing
_df = _df.copy()
_df = _df[_df['dyslexia_bin'].isin([0.0, 1.0])]

# Subset to dyslexic individuals
_df_dys = _df[_df['dyslexia_bin'] == 1.0].copy()

# Keep relevant columns and drop missing speed or reader_view
_df_dys = _df_dys.dropna(subset=['speed', 'reader_view'])

# Split groups
rv1 = _df_dys[_df_dys['reader_view'] == 1]['speed']
rv0 = _df_dys[_df_dys['reader_view'] == 0]['speed']

# Basic stats
stats_basic = {
    'n_dys_total': int(_df_dys.shape[0]),
    'n_rv1': int(rv1.shape[0]),
    'n_rv0': int(rv0.shape[0]),
    'mean_rv1': float(rv1.mean()),
    'mean_rv0': float(rv0.mean()),
    'median_rv1': float(rv1.median()),
    'median_rv0': float(rv0.median()),
    'std_rv1': float(rv1.std(ddof=1)),
    'std_rv0': float(rv0.std(ddof=1)),
}

# Effect size (Cohen's d)
# Pooled SD
n1, n0 = rv1.shape[0], rv0.shape[0]
if n1 > 1 and n0 > 1:
    s1 = rv1.var(ddof=1)
    s0 = rv0.var(ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n0 - 1) * s0) / (n1 + n0 - 2))
    cohens_d = (rv1.mean() - rv0.mean()) / s_pooled if s_pooled > 0 else np.nan
else:
    cohens_d = np.nan

# Welch t-test
if n1 > 1 and n0 > 1:
    t_stat, t_p = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
else:
    t_stat, t_p = np.nan, np.nan

# Mann-Whitney U test (two-sided)
if n1 > 0 and n0 > 0:
    try:
        u_stat, u_p = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    except ValueError:
        u_stat, u_p = np.nan, np.nan
else:
    u_stat, u_p = np.nan, np.nan

# Log-speed regression with controls
# Add small constant to avoid log(0)
_df_dys = _df_dys.copy()
_df_dys['log_speed'] = np.log(_df_dys['speed'] + 1)

# Use page_id and num_words as controls; include device as categorical
# Keep rows with needed data
reg_df = _df_dys.dropna(subset=['log_speed', 'reader_view', 'num_words', 'page_id', 'device'])

model = smf.ols('log_speed ~ reader_view + num_words + C(page_id) + C(device)', data=reg_df).fit(cov_type='HC3')

# Extract coefficient for reader_view
coef = model.params.get('reader_view', np.nan)
se = model.bse.get('reader_view', np.nan)
pval = model.pvalues.get('reader_view', np.nan)

# Convert log-speed effect to approximate percent change
pct_change = (np.exp(coef) - 1.0) * 100 if np.isfinite(coef) else np.nan

results = {
    'basic': stats_basic,
    'cohens_d': float(cohens_d),
    't_test_p': float(t_p),
    'mannwhitney_p': float(u_p),
    'reg_n': int(reg_df.shape[0]),
    'reg_coef': float(coef),
    'reg_se': float(se),
    'reg_p': float(pval),
    'reg_pct_change': float(pct_change),
    'reg_r2': float(model.rsquared),
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
