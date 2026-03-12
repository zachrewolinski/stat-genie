import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
df = pd.read_csv('reading.csv')

# Define dyslexic subset: primary uses dyslexia_bin==1
sub = df[df['dyslexia_bin'] == 1].copy()

# Basic descriptive stats by reader_view
summary = sub.groupby('reader_view')['speed'].agg(['count','mean','median','std'])

# Welch t-test
rv0 = sub[sub['reader_view'] == 0]['speed']
rv1 = sub[sub['reader_view'] == 1]['speed']

# Handle any missing
rv0 = rv0.dropna()
rv1 = rv1.dropna()

welch = stats.ttest_ind(rv1, rv0, equal_var=False)

# Effect size (Hedges g)
# pooled SD for two groups
n0, n1 = len(rv0), len(rv1)
mean0, mean1 = rv0.mean(), rv1.mean()
var0, var1 = rv0.var(ddof=1), rv1.var(ddof=1)
pooled = ((n0-1)*var0 + (n1-1)*var1) / (n0+n1-2)
cohen_d = (mean1 - mean0) / np.sqrt(pooled)
# Hedges correction
J = 1 - (3/(4*(n0+n1)-9))
hedges_g = J * cohen_d

# Nonparametric Mann-Whitney U
mw = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')

# Regression with cluster-robust SE by uuid on log(speed)
# Add small constant to avoid log(0)
sub['log_speed'] = np.log(sub['speed'])

# Remove infinities if any
sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=['log_speed'])

# Categorical controls: page_id
# Use statsmodels formula with C(page_id)
formula = 'log_speed ~ reader_view + C(page_id)'

model = smf.ols(formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['uuid']})

coef = model.params['reader_view']
se = model.bse['reader_view']
pval = model.pvalues['reader_view']

# Sensitivity: dyslexia severity (dyslexia in {1,2})
sub2 = df[df['dyslexia'].isin([1,2])].copy()
rv0b = sub2[sub2['reader_view']==0]['speed'].dropna()
rv1b = sub2[sub2['reader_view']==1]['speed'].dropna()
welch2 = stats.ttest_ind(rv1b, rv0b, equal_var=False)

print('SUMMARY')
print(summary)
print('\nWelch t-test (rv1 - rv0):', welch)
print('Hedges g:', hedges_g)
print('Mann-Whitney U:', mw)
print('\nClustered OLS on log_speed: coef', coef, 'se', se, 'p', pval)
print('\nSensitivity (dyslexia in {1,2}) Welch:', welch2)
