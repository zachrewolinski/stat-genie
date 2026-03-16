import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


df=pd.read_csv('reading.csv')

# map columns
# dyslexia status assumed in 'dyslexia' (0=no,1=dyslexia,2=severe)
# reader_view indicator assumed in 'language' (0/1)
# reading speed assumed in 'running_time'

# filter dyslexic participants (dyslexia > 0)
sub=df[(df['dyslexia'].notna()) & (df['dyslexia']>0)]

# keep valid rows
sub=sub[sub['language'].notna() & sub['running_time'].notna() & sub['speed'].notna()]

# basic counts
print('rows', len(sub))
print('unique participants', sub['speed'].nunique())
print('reader_view counts', sub['language'].value_counts())

# descriptive stats by reader_view
stats_by=sub.groupby('language')['running_time'].agg(['count','mean','median','std'])
print('\nSpeed by reader_view:')
print(stats_by)

# log-transform speed to reduce skew
sub = sub.copy()
sub['log_speed']=np.log(sub['running_time'])

# t-test on log speed
rv0=sub[sub['language']==0]['log_speed']
rv1=sub[sub['language']==1]['log_speed']

# Welch's t-test
tt=stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
print('\nWelch t-test log_speed rv1 vs rv0:', tt)

# effect size (Cohen d) on log_speed
mean_diff=rv1.mean()-rv0.mean()
pooled_sd=np.sqrt(((rv1.var(ddof=1)+rv0.var(ddof=1))/2))
d=mean_diff/pooled_sd
print('Cohen d (log_speed)', d)

# regression with cluster-robust SE by participant id
# model: log_speed ~ reader_view
model = smf.ols('log_speed ~ language', data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['speed']})
print('\nCluster-robust OLS:')
print(model.summary())

# also check on raw speed with cluster robust
model_raw = smf.ols('running_time ~ language', data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['speed']})
print('\nCluster-robust OLS (raw speed):')
print(model_raw.summary())

