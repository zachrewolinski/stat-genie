import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('reading.csv')

# Ensure columns
# Define dyslexia subset
# Use dyslexia_bin==1 as dyslexia (including severe)

df = _df.copy()

# Filter dyslexia participants
if 'dyslexia_bin' in df.columns:
    dys_df = df[df['dyslexia_bin'] == 1].copy()
else:
    # fallback: dyslexia > 0
    dys_df = df[df['dyslexia'] > 0].copy()

# Basic counts
n_total = len(df)
ndys = len(dys_df)

# Guard: remove missing speed or reader_view
cols_needed = ['speed', 'reader_view']
for col in cols_needed:
    if col not in dys_df.columns:
        raise ValueError(f"Missing {col}")

dys_df = dys_df.dropna(subset=cols_needed + ['uuid'])

# Convert reader_view to int
# Some values might be float
rv = dys_df['reader_view'].astype(int)

# Summary stats
speed = dys_df['speed'].astype(float)

# Use log speed due to skew
log_speed = np.log(speed)

# group stats
summary = dys_df.groupby('reader_view')['speed'].agg(['count','mean','median','std']).rename_axis('reader_view')
summary_log = dys_df.groupby('reader_view').apply(lambda g: pd.Series({
    'count': g['speed'].shape[0],
    'mean_log_speed': np.log(g['speed']).mean(),
    'std_log_speed': np.log(g['speed']).std(),
}))

# T-test on log speed (Welch)
rv0 = log_speed[rv == 0]
rv1 = log_speed[rv == 1]

# If one group empty, handle
if len(rv0) > 1 and len(rv1) > 1:
    tstat, pval = stats.ttest_ind(rv1, rv0, equal_var=False)
else:
    tstat, pval = np.nan, np.nan

# Effect size: Cohen's d on log speed
# d = (mean1-mean0)/pooled sd
if len(rv0) > 1 and len(rv1) > 1:
    mean0, mean1 = rv0.mean(), rv1.mean()
    sd0, sd1 = rv0.std(ddof=1), rv1.std(ddof=1)
    pooled_sd = np.sqrt(((len(rv0)-1)*sd0**2 + (len(rv1)-1)*sd1**2) / (len(rv0)+len(rv1)-2))
    d_log = (mean1 - mean0) / pooled_sd if pooled_sd > 0 else np.nan
else:
    d_log = np.nan

# Cluster-robust OLS on log_speed with reader_view controlling for page, num_words, device? Use minimal controls
# We can include fixed effects for page_id and num_words, language maybe. We'll do: reader_view + page_id + num_words + device
# Keep simple to avoid multicollinearity. Use categorical dummies.

# Prepare dataframe
reg_df = dys_df[['uuid', 'reader_view', 'speed', 'page_id', 'num_words', 'device']].dropna().copy()
reg_df['log_speed'] = np.log(reg_df['speed'].astype(float))

# Build design matrix
X = pd.get_dummies(reg_df[['reader_view','page_id','device']], drop_first=True)
# Add num_words numeric if present
X['num_words'] = reg_df['num_words'].astype(float)
X = sm.add_constant(X, has_constant='add')

model = sm.OLS(reg_df['log_speed'], X)
# cluster by uuid
res = model.fit(cov_type='cluster', cov_kwds={'groups': reg_df['uuid']})

# Also compute per-participant mean difference to reduce dependence
# Compute average log speed per uuid per reader_view, then compare within-participant if both conditions exist
pp = reg_df[['uuid','reader_view','log_speed']].copy()
mean_by = pp.groupby(['uuid','reader_view'])['log_speed'].mean().reset_index()
# pivot to wide
pivot = mean_by.pivot(index='uuid', columns='reader_view', values='log_speed')
# participants with both
paired = pivot.dropna()

if len(paired) > 1:
    # paired t-test
    t_paired, p_paired = stats.ttest_rel(paired[1], paired[0])
    mean_diff = (paired[1] - paired[0]).mean()
    sd_diff = (paired[1] - paired[0]).std(ddof=1)
    d_paired = mean_diff / sd_diff if sd_diff > 0 else np.nan
else:
    t_paired, p_paired, mean_diff, d_paired = np.nan, np.nan, np.nan, np.nan

# Output results
print('N total', n_total)
print('N dyslexia rows', ndys)
print('Group summary speed (raw):')
print(summary)
print('Group summary log speed:')
print(summary_log)
print('Welch t-test on log speed: t=', tstat, 'p=', pval, 'd_log=', d_log)
print('OLS cluster on log speed:')
print(res.summary().tables[1])
print('reader_view coef:', res.params.get('reader_view', np.nan), 'p=', res.pvalues.get('reader_view', np.nan))
print('Paired within-uuid t-test (mean log speed): t=', t_paired, 'p=', p_paired, 'mean_diff=', mean_diff, 'd_paired=', d_paired, 'n_pairs=', len(paired))
