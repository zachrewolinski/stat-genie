import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('reading.csv')

# Basic cleaning
# Ensure numeric columns
num_cols = [
    'reader_view','running_time','adjusted_running_time','scrolling_time',
    'num_words','correct_rate','img_width','age','dyslexia','retake_trial',
    'dyslexia_bin','Flesch_Kincaid','speed'
]
for c in num_cols:
    if c in _df.columns:
        _df[c] = pd.to_numeric(_df[c], errors='coerce')

# Define dyslexia group (includes severe)
# Prefer dyslexia_bin when available, else dyslexia>0
if 'dyslexia_bin' in _df.columns:
    dys_mask = _df['dyslexia_bin'] == 1
else:
    dys_mask = _df['dyslexia'] > 0

# Filter to dyslexic participants and valid reader_view/speed
_df_dys = _df.loc[dys_mask].copy()
_df_dys = _df_dys.loc[_df_dys['reader_view'].isin([0,1])]
_df_dys = _df_dys.loc[_df_dys['speed'] > 0]

# Counts
counts = _df_dys['reader_view'].value_counts(dropna=False).to_dict()

# Group stats
stats_by_group = _df_dys.groupby('reader_view')['speed'].agg(['count','mean','median','std']).to_dict()

# T-test (Welch)
rv1 = _df_dys.loc[_df_dys['reader_view'] == 1, 'speed']
rv0 = _df_dys.loc[_df_dys['reader_view'] == 0, 'speed']
welch = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')

# Mann-Whitney U (non-parametric)
try:
    mwu = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
except Exception:
    mwu = None

# Effect size (Cohen's d) on raw speed
mean1 = rv1.mean()
mean0 = rv0.mean()
std1 = rv1.std(ddof=1)
std0 = rv0.std(ddof=1)
n1 = rv1.shape[0]
n0 = rv0.shape[0]
pooled = np.sqrt(((n1-1)*std1**2 + (n0-1)*std0**2) / (n1+n0-2)) if (n1+n0-2) > 0 else np.nan
cohen_d = (mean1 - mean0) / pooled if pooled and pooled > 0 else np.nan

# Log-speed analysis to reduce outliers
_df_dys['log_speed'] = np.log(_df_dys['speed'])
rv1_log = _df_dys.loc[_df_dys['reader_view'] == 1, 'log_speed']
rv0_log = _df_dys.loc[_df_dys['reader_view'] == 0, 'log_speed']
welch_log = stats.ttest_ind(rv1_log, rv0_log, equal_var=False, nan_policy='omit')

# Regression with cluster-robust SEs by participant
# Prepare categorical variables (fill missing)
cat_cols = ['page_id','device','education','gender','language','english_native']
for c in cat_cols:
    if c in _df_dys.columns:
        _df_dys[c] = _df_dys[c].astype('object').fillna('missing')

# Build formula with a modest set of controls
formula = (
    'log_speed ~ reader_view + num_words + Flesch_Kincaid + age + '
    'correct_rate + retake_trial + C(page_id) + C(device) + C(english_native)'
)

# Drop rows with missing values in model variables or uuid to keep clustering aligned
model_vars = ['log_speed', 'reader_view', 'num_words', 'Flesch_Kincaid', 'age',
              'correct_rate', 'retake_trial', 'page_id', 'device', 'english_native', 'uuid']
_df_model = _df_dys.dropna(subset=model_vars).copy()

model = smf.ols(formula, data=_df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': _df_model['uuid']}
)

# Extract coefficient info for reader_view
coef = model.params.get('reader_view', np.nan)
se = model.bse.get('reader_view', np.nan)
pval = model.pvalues.get('reader_view', np.nan)

# Package results
results = {
    'n_dys': int(_df_dys.shape[0]),
    'counts_reader_view': {str(k): int(v) for k, v in counts.items()},
    'speed_stats_by_reader_view': stats_by_group,
    'welch_t': {'stat': float(welch.statistic), 'p': float(welch.pvalue)},
    'mwu': None if mwu is None else {'stat': float(mwu.statistic), 'p': float(mwu.pvalue)},
    'cohen_d': float(cohen_d),
    'welch_t_log': {'stat': float(welch_log.statistic), 'p': float(welch_log.pvalue)},
    'reg_reader_view': {'coef': float(coef), 'se': float(se), 'p': float(pval)},
    'reg_nobs': int(model.nobs),
    'reg_r2': float(model.rsquared)
}

print(json.dumps(results, indent=2))
