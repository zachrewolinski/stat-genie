import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Identify dyslexic participants
# feature17: 1 dyslexia, 0 no dyslexia
# feature12: 0 no dyslexia, 1 dyslexia, 2 severe dyslexia

# Use feature17 as primary indicator; fall back to feature12 when feature17 is missing

df['dyslexia_flag'] = df['feature17']
missing_mask = df['dyslexia_flag'].isna()
df.loc[missing_mask, 'dyslexia_flag'] = df.loc[missing_mask, 'feature12'].apply(lambda x: 1.0 if x in [1.0, 2.0] else (0.0 if x == 0.0 else np.nan))

# subset dyslexic
dys = df[df['dyslexia_flag'] == 1.0].copy()

# Reader view indicator
# feature3: 1 activated, 0 not

# reading speed
speed = 'feature20'

# Group statistics
stats_by_group = dys.groupby('feature3')[speed].agg(['count','mean','median','std'])

# Welch t-test
rv = dys[dys['feature3'] == 1][speed]
no_rv = dys[dys['feature3'] == 0][speed]

# Ensure non-empty groups
if len(rv) == 0 or len(no_rv) == 0:
    raise SystemExit('One of the groups is empty; cannot compare.')

t_stat, t_p = stats.ttest_ind(rv, no_rv, equal_var=False, nan_policy='omit')

# Mann-Whitney U (nonparametric)
try:
    u_stat, u_p = stats.mannwhitneyu(rv, no_rv, alternative='two-sided')
except ValueError:
    u_stat, u_p = np.nan, np.nan

# Effect size: Cohen's d (Welch-style pooled SD)
rv_mean = rv.mean()
no_mean = no_rv.mean()
rv_var = rv.var(ddof=1)
no_var = no_rv.var(ddof=1)
pooled_sd = np.sqrt((rv_var + no_var) / 2)
cohen_d = (rv_mean - no_mean) / pooled_sd if pooled_sd > 0 else np.nan

# Robust analysis: log1p transform and mixed effects with participant random intercept
# MixedLM can be sensitive; wrap in try

# add log speed

dys['log_speed'] = np.log1p(dys[speed])

mixed_result = None
mixed_p = None
mixed_coef = None

try:
    model = smf.mixedlm('log_speed ~ feature3', data=dys, groups=dys['feature1'])
    mixed_result = model.fit(reml=False, method='lbfgs')
    mixed_coef = mixed_result.params.get('feature3', np.nan)
    mixed_p = mixed_result.pvalues.get('feature3', np.nan)
except Exception:
    mixed_result = None

# Summaries
summary = {
    'n_total_dyslexia': int(len(dys)),
    'n_reader_view': int(len(rv)),
    'n_no_reader_view': int(len(no_rv)),
    'group_stats': stats_by_group.to_dict(),
    't_test': {'t': float(t_stat), 'p': float(t_p)},
    'mannwhitney': {'u': float(u_stat) if not np.isnan(u_stat) else None, 'p': float(u_p) if not np.isnan(u_p) else None},
    'cohen_d': float(cohen_d),
    'mixedlm': {'coef_log': float(mixed_coef) if mixed_coef is not None else None, 'p': float(mixed_p) if mixed_p is not None else None}
}

# Save summary for inspection
pd.Series(summary).to_json('analysis_summary.json')

# Print key results
print(summary)
