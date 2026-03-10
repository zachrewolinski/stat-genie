import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

path = 'reading.csv'
df = pd.read_csv(path)

# focus on dyslexia individuals
dys = df[df['dyslexia_bin'] == 1].copy()

# basic counts
n_rows = len(dys)
participants = dys['uuid'].nunique()

# group stats
by_rv = dys.groupby('reader_view')['speed'].agg(['count','mean','median','std'])

# log speed
# Avoid issues if any non-positive
if (dys['speed'] <= 0).any():
    dys = dys[dys['speed'] > 0]

dys['log_speed'] = np.log(dys['speed'])

# Welch t-test on log speed (row-level)
rv1 = dys[dys['reader_view'] == 1]['log_speed']
rv0 = dys[dys['reader_view'] == 0]['log_speed']
welch = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')

# per-uuid averages for paired test (only those with both conditions)
avg = dys.groupby(['uuid','reader_view'])['log_speed'].mean().unstack()
paired = avg.dropna()
paired_t = stats.ttest_rel(paired[1], paired[0], nan_policy='omit') if len(paired) > 1 else None

# Regression with cluster-robust SE by uuid
# Use modest controls to avoid overfitting; drop rows with missing covariates
formula = 'log_speed ~ reader_view + num_words + Flesch_Kincaid + C(page_id) + C(device) + retake_trial'
model_df = dys[['log_speed','reader_view','num_words','Flesch_Kincaid','page_id','device','retake_trial','uuid']].dropna()
model = smf.ols(formula, data=model_df).fit(cov_type='cluster', cov_kwds={'groups': model_df['uuid']})

beta = model.params.get('reader_view', np.nan)
pval = model.pvalues.get('reader_view', np.nan)

# percent change from log
pct = (np.exp(beta) - 1) * 100 if np.isfinite(beta) else np.nan

print('Rows (dyslexia):', n_rows)
print('Participants (dyslexia):', participants)
print('\nGroup stats (speed):')
print(by_rv)
print('\nWelch t-test log_speed: statistic=%.4f p=%.6f' % (welch.statistic, welch.pvalue))
print('\nPaired t-test log_speed (per-uuid avg): n=%d stat=%s p=%s' % (
    len(paired),
    '%.4f' % paired_t.statistic if paired_t is not None else 'NA',
    '%.6f' % paired_t.pvalue if paired_t is not None else 'NA'
))
print('\nCluster-robust OLS:')
print('beta(reader_view)=%.6f p=%.6f pct_change=%.2f%%' % (beta, pval, pct))
print(model.summary().tables[1])
