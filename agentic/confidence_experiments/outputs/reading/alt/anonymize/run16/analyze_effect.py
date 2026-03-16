import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


df = pd.read_csv('reading.csv')

# Filter to dyslexic individuals (feature17 == 1)
df = df[df['feature17'] == 1].copy()

# Keep relevant columns and drop missing
cols = ['feature1','feature3','feature20','feature7','feature19','feature11','feature15','feature2','feature16']

df = df[cols].dropna(subset=['feature3','feature20'])

# Basic counts
n_total = len(df)
counts = df['feature3'].value_counts().sort_index()
print('n_total', n_total)
print('counts by reader view', counts.to_dict())

# split groups
rv0 = df[df['feature3'] == 0]['feature20']
rv1 = df[df['feature3'] == 1]['feature20']

# Welch t-test
welch = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')

# Mann-Whitney U
mw = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')

# effect size (Cohen's d)
mean0, mean1 = rv0.mean(), rv1.mean()
std0, std1 = rv0.std(ddof=1), rv1.std(ddof=1)
# pooled SD for unequal sizes
n0, n1 = rv0.shape[0], rv1.shape[0]
pooled_sd = np.sqrt(((n0-1)*std0**2 + (n1-1)*std1**2) / (n0+n1-2))
cohen_d = (mean1 - mean0) / pooled_sd if pooled_sd > 0 else np.nan

print('mean0', mean0, 'mean1', mean1, 'diff', mean1-mean0)
print('median0', rv0.median(), 'median1', rv1.median())
print('welch', welch)
print('mannwhitney', mw)
print('cohen_d', cohen_d)

# log transform to reduce skew
rv0_log = np.log1p(rv0)
rv1_log = np.log1p(rv1)
welch_log = stats.ttest_ind(rv1_log, rv0_log, equal_var=False, nan_policy='omit')
print('welch_log', welch_log)

# Paired analysis for participants who experienced both conditions
pivot = df.pivot_table(index='feature1', columns='feature3', values='feature20', aggfunc='mean')
paired = pivot.dropna()
print('paired participants', paired.shape[0])
if paired.shape[0] > 1:
    t_paired = stats.ttest_rel(paired[1], paired[0], nan_policy='omit')
    diff = paired[1] - paired[0]
    dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
    print('paired mean diff', diff.mean(), 'median diff', diff.median())
    print('paired t', t_paired, 'cohen_dz', dz)

# Regression with participant fixed effects and controls
# Use log speed to reduce skew
# Drop rows with missing covariates
reg_df = df.dropna(subset=['feature7','feature19','feature11','feature15','feature2','feature16']).copy()
reg_df['log_speed'] = np.log1p(reg_df['feature20'])

# Fit OLS with participant fixed effects (C(feature1))
formula = 'log_speed ~ feature3 + feature7 + feature19 + C(feature11) + C(feature15) + C(feature2) + feature16 + C(feature1)'
model = smf.ols(formula, data=reg_df).fit()
robust = model.get_robustcov_results(cov_type='cluster', groups=reg_df['feature1'])

# Extract coefficient and p-value for feature3
exog_names = robust.model.exog_names
if 'feature3' in exog_names:
    idx = exog_names.index('feature3')
    coef = robust.params[idx]
    pval = robust.pvalues[idx]
    print('regression coef feature3', coef, 'p', pval)
else:
    print('feature3 not in regression design matrix')

