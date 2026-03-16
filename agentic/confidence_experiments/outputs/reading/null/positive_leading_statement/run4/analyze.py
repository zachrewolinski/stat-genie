import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = 'reading.csv'

df = pd.read_csv(path)

# Basic cleaning
# Keep valid speed > 0, reader_view in {0,1}
df = df[(df['speed'] > 0) & (df['reader_view'].isin([0,1]))]

# Define dyslexia subset (binary indicator)
if 'dyslexia_bin' in df.columns:
    dys_df = df[df['dyslexia_bin'] == 1].copy()
else:
    dys_df = df[df['dyslexia'].isin([1,2])].copy()

# Log-transform speed to reduce skew
# Add small epsilon if needed (speed > 0 already)
dys_df['log_speed'] = np.log(dys_df['speed'])

# Group summaries
summary = dys_df.groupby('reader_view')['speed'].agg(['count','mean','median','std'])
log_summary = dys_df.groupby('reader_view')['log_speed'].agg(['count','mean','std'])

# Welch t-test on raw speed
rv0 = dys_df[dys_df['reader_view'] == 0]['speed']
rv1 = dys_df[dys_df['reader_view'] == 1]['speed']

welch_raw = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')

# Welch t-test on log speed
lrv0 = dys_df[dys_df['reader_view'] == 0]['log_speed']
lrv1 = dys_df[dys_df['reader_view'] == 1]['log_speed']

welch_log = stats.ttest_ind(lrv1, lrv0, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (nonparametric)
# Use alternative='two-sided'
try:
    mw = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
except Exception:
    mw = None

# Regression with controls and cluster-robust SE by uuid
# Use log_speed as outcome; controls for page length/readability
# Avoid missing values in controls
reg_df = dys_df.dropna(subset=['log_speed', 'reader_view', 'num_words', 'Flesch_Kincaid', 'page_id', 'uuid']).copy()

# Ensure categorical page_id
model = smf.ols('log_speed ~ reader_view + num_words + Flesch_Kincaid + C(page_id)', data=reg_df)
fit = model.fit()

# Cluster-robust SE by uuid
fit_cluster = fit.get_robustcov_results(cov_type='cluster', groups=reg_df['uuid'])

# Extract effect of reader_view with safe indexing
params = pd.Series(fit_cluster.params, index=fit_cluster.model.exog_names)
ses = pd.Series(fit_cluster.bse, index=fit_cluster.model.exog_names)
pvals = pd.Series(fit_cluster.pvalues, index=fit_cluster.model.exog_names)

coef = params['reader_view']
se = ses['reader_view']
# 95% CI
ci_low = coef - 1.96*se
ci_high = coef + 1.96*se

# Convert log-effect to percent change
pct = (np.exp(coef) - 1) * 100
pct_low = (np.exp(ci_low) - 1) * 100
pct_high = (np.exp(ci_high) - 1) * 100

# Save results to a dict for easier inspection
results = {
    'n_dyslexia': int(len(dys_df)),
    'summary_speed': summary,
    'summary_log_speed': log_summary,
    'welch_raw_t': float(welch_raw.statistic),
    'welch_raw_p': float(welch_raw.pvalue),
    'welch_log_t': float(welch_log.statistic),
    'welch_log_p': float(welch_log.pvalue),
    'mannwhitney_u': None if mw is None else float(mw.statistic),
    'mannwhitney_p': None if mw is None else float(mw.pvalue),
    'reg_coef_log': float(coef),
    'reg_se_log': float(se),
    'reg_p': float(pvals['reader_view']),
    'reg_ci_log': (float(ci_low), float(ci_high)),
    'reg_pct_effect': float(pct),
    'reg_pct_ci': (float(pct_low), float(pct_high)),
    'reg_n': int(fit_cluster.nobs)
}

print('Dyslexia subset size:', results['n_dyslexia'])
print('\nSpeed summary by reader_view (raw):')
print(summary)
print('\nLog-speed summary by reader_view:')
print(log_summary)
print('\nWelch t-test raw speed: t=%.4f, p=%.4g' % (results['welch_raw_t'], results['welch_raw_p']))
print('Welch t-test log speed: t=%.4f, p=%.4g' % (results['welch_log_t'], results['welch_log_p']))
if mw is not None:
    print('Mann-Whitney U: U=%.4f, p=%.4g' % (results['mannwhitney_u'], results['mannwhitney_p']))
print('\nRegression (log_speed ~ reader_view + controls, cluster-robust by uuid):')
print('coef=%.6f (SE=%.6f), p=%.4g' % (results['reg_coef_log'], results['reg_se_log'], results['reg_p']))
print('95%% CI log coef: [%.6f, %.6f]' % results['reg_ci_log'])
print('Approx %% effect: %.2f%% (95%% CI [%.2f%%, %.2f%%])' % (results['reg_pct_effect'], results['reg_pct_ci'][0], results['reg_pct_ci'][1]))

# Also compute simple difference in means and medians
mean_diff = summary.loc[1, 'mean'] - summary.loc[0, 'mean'] if 0 in summary.index and 1 in summary.index else np.nan
median_diff = summary.loc[1, 'median'] - summary.loc[0, 'median'] if 0 in summary.index and 1 in summary.index else np.nan
print('\nMean speed difference (reader_view=1 - 0):', mean_diff)
print('Median speed difference (reader_view=1 - 0):', median_diff)
