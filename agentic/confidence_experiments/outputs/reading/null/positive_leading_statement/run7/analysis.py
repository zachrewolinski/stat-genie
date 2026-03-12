import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

DF = pd.read_csv('reading.csv')

# Filter to dyslexia individuals
sub = DF[DF['dyslexia_bin'] == 1].copy()
sub = sub.dropna(subset=['reader_view', 'speed', 'uuid'])

# Basic group stats
rv0 = sub[sub['reader_view'] == 0]['speed']
rv1 = sub[sub['reader_view'] == 1]['speed']

summary = {
    'n_total': len(sub),
    'n_rv0': len(rv0),
    'n_rv1': len(rv1),
    'mean_rv0': rv0.mean(),
    'mean_rv1': rv1.mean(),
    'median_rv0': rv0.median(),
    'median_rv1': rv1.median(),
}

# Welch t-test
welch = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (two-sided)
try:
    mw = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
except Exception:
    mw = None

# Effect size (Cohen's d for independent samples)
# Using pooled SD (unbiased)
rv0_var = rv0.var(ddof=1)
rv1_var = rv1.var(ddof=1)
pooled_sd = np.sqrt(((len(rv0)-1)*rv0_var + (len(rv1)-1)*rv1_var) / (len(rv0)+len(rv1)-2))
cohen_d = (rv1.mean() - rv0.mean()) / pooled_sd if pooled_sd > 0 else np.nan

# Paired within-subject analysis for participants with both conditions
pivot = sub.pivot_table(index='uuid', columns='reader_view', values='speed', aggfunc='mean')
paired = pivot.dropna(subset=[0, 1])
paired_diff = paired[1] - paired[0]
paired_t = stats.ttest_rel(paired[1], paired[0], nan_policy='omit')
# Wilcoxon signed-rank (if at least some non-zero diffs)
if (paired_diff != 0).any():
    try:
        wsr = stats.wilcoxon(paired[1], paired[0])
    except Exception:
        wsr = None
else:
    wsr = None

# Regression with log(speed)
sub = sub.copy()
sub['log_speed'] = np.log(sub['speed'])
# Keep rows with needed covariates
reg = sub.dropna(subset=['log_speed', 'reader_view', 'num_words', 'page_id', 'device', 'retake_trial'])

# Use a modest model to avoid overfitting
formula = 'log_speed ~ reader_view + num_words + C(page_id) + C(device) + retake_trial'
model = smf.ols(formula, data=reg).fit(cov_type='cluster', cov_kwds={'groups': reg['uuid']})

# Extract reader_view coefficient
coef = model.params.get('reader_view', np.nan)
pval = model.pvalues.get('reader_view', np.nan)

# Summarize
print('SUMMARY')
print(summary)
print('Welch t-test: statistic=%.4f p=%.4g' % (welch.statistic, welch.pvalue))
if mw:
    print('Mann-Whitney U: statistic=%.4f p=%.4g' % (mw.statistic, mw.pvalue))
print('Cohen d: %.4f' % cohen_d)
print('Paired sample count (both conditions):', len(paired))
print('Paired t-test: statistic=%.4f p=%.4g' % (paired_t.statistic, paired_t.pvalue))
if wsr:
    print('Wilcoxon signed-rank: statistic=%.4f p=%.4g' % (wsr.statistic, wsr.pvalue))
print('Regression (log_speed) reader_view coef=%.4f p=%.4g' % (coef, pval))
