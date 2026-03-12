import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('reading.csv')

# Harmonize dyslexia indicator
if 'dyslexia_bin' in _df.columns:
    _df['is_dyslexic'] = _df['dyslexia_bin'] == 1
elif 'dyslexia' in _df.columns:
    _df['is_dyslexic'] = _df['dyslexia'].fillna(0) > 0
else:
    raise ValueError('No dyslexia indicator found')

# Keep necessary columns
cols_needed = ['uuid', 'reader_view', 'speed', 'num_words', 'Flesch_Kincaid', 'page_id', 'device', 'language']
for c in cols_needed:
    if c not in _df.columns:
        _df[c] = np.nan

# Filter dyslexic participants
_df = _df[_df['is_dyslexic']].copy()

# Clean
_df = _df.dropna(subset=['reader_view', 'speed'])
_df = _df[_df['speed'] > 0]

# Ensure reader_view is binary
_df['reader_view'] = _df['reader_view'].astype(int)

# Log speed for skew
_df['log_speed'] = np.log(_df['speed'])

# Summary counts
n_total = len(_df)
counts = _df['reader_view'].value_counts().to_dict()

# Group means
mean_speed = _df.groupby('reader_view')['speed'].mean()
mean_log_speed = _df.groupby('reader_view')['log_speed'].mean()

# Welch t-test (independent)
rv1 = _df[_df['reader_view'] == 1]['speed']
rv0 = _df[_df['reader_view'] == 0]['speed']
welch_t = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')

rv1_log = _df[_df['reader_view'] == 1]['log_speed']
rv0_log = _df[_df['reader_view'] == 0]['log_speed']
welch_t_log = stats.ttest_ind(rv1_log, rv0_log, equal_var=False, nan_policy='omit')

# Effect size (Cohen's d) on log scale
mean_diff_log = mean_log_speed.get(1, np.nan) - mean_log_speed.get(0, np.nan)
std_pooled = np.sqrt(((rv1_log.var(ddof=1) + rv0_log.var(ddof=1)) / 2))
cohen_d_log = mean_diff_log / std_pooled if std_pooled and not np.isnan(std_pooled) else np.nan

# Percent difference from log means
pct_diff = (np.exp(mean_diff_log) - 1) * 100 if not np.isnan(mean_diff_log) else np.nan

# Paired analysis: participants with both conditions
paired = (
    _df.groupby(['uuid', 'reader_view'])['log_speed']
    .mean()
    .unstack('reader_view')
    .dropna()
)
paired_t = None
paired_diff_mean = None
paired_pct = None
if not paired.empty:
    diff = paired[1] - paired[0]
    paired_diff_mean = diff.mean()
    paired_pct = (np.exp(paired_diff_mean) - 1) * 100
    paired_t = stats.ttest_1samp(diff, 0.0, nan_policy='omit')

# Regression with controls + clustered SEs by uuid
# Use only rows with required columns
reg_df = _df.copy()
# Handle missing controls by filling with mean/mode; keep indicator for missing
for col in ['num_words', 'Flesch_Kincaid']:
    if reg_df[col].isna().any():
        reg_df[col] = reg_df[col].fillna(reg_df[col].mean())

# Convert categorical
for col in ['page_id', 'device', 'language']:
    if col in reg_df.columns:
        reg_df[col] = reg_df[col].astype('category')

# Build formula
formula = 'log_speed ~ reader_view'
if 'num_words' in reg_df.columns:
    formula += ' + num_words'
if 'Flesch_Kincaid' in reg_df.columns:
    formula += ' + Flesch_Kincaid'
if 'page_id' in reg_df.columns:
    formula += ' + C(page_id)'
if 'device' in reg_df.columns:
    formula += ' + C(device)'
if 'language' in reg_df.columns:
    formula += ' + C(language)'

model = smf.ols(formula, data=reg_df)
try:
    res = model.fit(cov_type='cluster', cov_kwds={'groups': reg_df['uuid']})
except Exception:
    res = model.fit()

coef = res.params.get('reader_view', np.nan)
pval = res.pvalues.get('reader_view', np.nan)

# Save results to json for inspection
results = {
    'n_total': int(n_total),
    'counts': {str(k): int(v) for k, v in counts.items()},
    'mean_speed': {str(k): float(v) for k, v in mean_speed.items()},
    'mean_log_speed': {str(k): float(v) for k, v in mean_log_speed.items()},
    'welch_t_speed': {'stat': float(welch_t.statistic), 'pvalue': float(welch_t.pvalue)},
    'welch_t_log': {'stat': float(welch_t_log.statistic), 'pvalue': float(welch_t_log.pvalue)},
    'cohen_d_log': float(cohen_d_log) if not np.isnan(cohen_d_log) else None,
    'pct_diff_log': float(pct_diff) if not np.isnan(pct_diff) else None,
    'paired_n': int(len(paired)),
    'paired_t': None if paired_t is None else {'stat': float(paired_t.statistic), 'pvalue': float(paired_t.pvalue)},
    'paired_pct': float(paired_pct) if paired_pct is not None else None,
    'reg_coef': float(coef) if not np.isnan(coef) else None,
    'reg_pvalue': float(pval) if not np.isnan(pval) else None,
}

print(json.dumps(results, indent=2))
