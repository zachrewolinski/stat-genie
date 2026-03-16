import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Variables
reader_view = 'feature3'  # 1 = reader view
reading_speed = 'feature20'  # assumed reading speed
# dyslexia indicators
binary_dyslexia = 'feature17'  # 1 = dyslexia
severity_dyslexia = 'feature12'  # 0/1/2

# Subset to dyslexia (binary)
sub = df[df[binary_dyslexia] == 1].copy()

# Basic counts
counts = sub[reader_view].value_counts().to_dict()

# Drop missing/invalid speeds
sub = sub[np.isfinite(sub[reading_speed])]

# Group stats
stats_by = sub.groupby(reader_view)[reading_speed].agg(['count','mean','median','std'])

# Welch t-test on log1p speed
rv1 = sub[sub[reader_view] == 1][reading_speed]
rv0 = sub[sub[reader_view] == 0][reading_speed]

log_rv1 = np.log1p(rv1)
log_rv0 = np.log1p(rv0)

tstat, pval = stats.ttest_ind(log_rv1, log_rv0, equal_var=False, nan_policy='omit')

# Mann-Whitney U test on raw
u_stat, u_pval = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')

# Effect size (Cohen's d) on log scale
n1, n0 = log_rv1.size, log_rv0.size
mean1, mean0 = log_rv1.mean(), log_rv0.mean()
var1, var0 = log_rv1.var(ddof=1), log_rv0.var(ddof=1)
pooled_sd = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1+n0-2)) if (n1+n0-2) > 0 else np.nan
cohen_d = (mean1 - mean0) / pooled_sd if pooled_sd > 0 else np.nan

# Regression controlling for word count, device, age, comprehension, language, page id
# Use log1p to reduce skew
sub = sub.copy()
sub['log_speed'] = np.log1p(sub[reading_speed])

# Build formula with categorical variables
formula = 'log_speed ~ C(feature3) + feature7 + feature10 + feature8 + C(feature11) + C(feature15) + C(feature2)'

model = smf.ols(formula, data=sub).fit()

# Extract coefficient for reader view (C(feature3)[T.1])
coef = model.params.get('C(feature3)[T.1]', np.nan)
coef_p = model.pvalues.get('C(feature3)[T.1]', np.nan)

# Also analyze using severity (feature12) for dyslexia/severe
sub_sev = df[df[severity_dyslexia] >= 1].copy()
sub_sev = sub_sev[np.isfinite(sub_sev[reading_speed])]
rv1s = sub_sev[sub_sev[reader_view]==1][reading_speed]
rv0s = sub_sev[sub_sev[reader_view]==0][reading_speed]
log_rv1s = np.log1p(rv1s)
log_rv0s = np.log1p(rv0s)

tstat_s, pval_s = stats.ttest_ind(log_rv1s, log_rv0s, equal_var=False, nan_policy='omit')

# Summary output
print('Dyslexia (feature17==1) counts by reader_view:', counts)
print('Group stats (speed):')
print(stats_by)
print('Welch t-test on log1p: t=', tstat, 'p=', pval)
print('Mann-Whitney U on raw: U=', u_stat, 'p=', u_pval)
print('Cohen d (log1p):', cohen_d)
print('Regression coef reader_view (log1p):', coef, 'p=', coef_p, 'R2=', model.rsquared)
print('---')
print('Severity dyslexia (feature12>=1) Welch t-test log1p: t=', tstat_s, 'p=', pval_s)
