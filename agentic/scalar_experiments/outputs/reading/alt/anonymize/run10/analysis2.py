import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('reading.csv')

# Define columns
speed_col = 'feature20'
reader_view_col = 'feature3'
dyslexia_col = 'feature17'
participant_col = 'feature1'
page_col = 'feature2'

# Subset to dyslexic participants (feature17 == 1)
sub = _df[_df[dyslexia_col] == 1].copy()

# Drop rows with missing speed or reader view
sub = sub.dropna(subset=[speed_col, reader_view_col, participant_col, page_col])

print('Total rows (dyslexia):', len(sub))
print('Participants (dyslexia):', sub[participant_col].nunique())
print('Reader view counts:', sub[reader_view_col].value_counts())

# Group stats
stats_by_group = sub.groupby(reader_view_col)[speed_col].agg(['count','mean','std','median'])
print(stats_by_group)

# Welch t-test
rv1 = sub[sub[reader_view_col]==1][speed_col]
rv0 = sub[sub[reader_view_col]==0][speed_col]

tstat, pval = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
print('Welch t-test', tstat, pval)

# Mann-Whitney U (two-sided)
try:
    ustat, pval_u = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
    print('Mann-Whitney U', ustat, pval_u)
except Exception as e:
    print('Mann-Whitney failed', e)

# Cohen's d
n1, n0 = len(rv1), len(rv0)
mean1, mean0 = rv1.mean(), rv0.mean()
std1, std0 = rv1.std(ddof=1), rv0.std(ddof=1)
pooled = np.sqrt(((n1-1)*std1**2 + (n0-1)*std0**2)/(n1+n0-2))
cohen_d = (mean1-mean0)/pooled
print('Cohen d', cohen_d)

# Mixed effects model: random intercept for participant
# Use log speed to reduce skew? We'll try both raw and log.
sub['log_speed'] = np.log(sub[speed_col])

# MixedLM raw
try:
    md = smf.mixedlm(f"{speed_col} ~ {reader_view_col}", sub, groups=sub[participant_col])
    mdf = md.fit(reml=False)
    print('MixedLM raw', mdf.summary())
except Exception as e:
    print('MixedLM raw failed', e)

# MixedLM log
try:
    md2 = smf.mixedlm(f"log_speed ~ {reader_view_col}", sub, groups=sub[participant_col])
    mdf2 = md2.fit(reml=False)
    print('MixedLM log', mdf2.summary())
except Exception as e:
    print('MixedLM log failed', e)

# OLS with participant clustered SE and page fixed effects
# Use categorical for page
sub['page'] = sub[page_col].astype('category')
ols = smf.ols(f"{speed_col} ~ {reader_view_col} + C(page)", data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub[participant_col]})
print('OLS cluster', ols.summary())

# log speed OLS
ols_log = smf.ols(f"log_speed ~ {reader_view_col} + C(page)", data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub[participant_col]})
print('OLS log cluster', ols_log.summary())

# Report coefficient, p-values
print('OLS coef (raw)', ols.params[reader_view_col], 'p', ols.pvalues[reader_view_col])
print('OLS coef (log)', ols_log.params[reader_view_col], 'p', ols_log.pvalues[reader_view_col])

