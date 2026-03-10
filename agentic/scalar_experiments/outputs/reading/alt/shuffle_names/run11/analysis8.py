import pandas as pd
import numpy as np
from scipy import stats

path='reading.csv'
df=pd.read_csv(path)

# define variables
reader_view = df['language']  # 0/1
# dyslexia status (0 no,1 dyslexia,2 severe) -> from 'device' column
# treat 1 or 2 as dyslexia

dyslexia_status = df['device']

# reading speed in words per minute from num_words and running_time (seconds)
wpm = df['num_words'] / df['running_time'] * 60

# filter dyslexic participants (1 or 2)
mask = dyslexia_status.isin([1,2]) & reader_view.notna() & wpm.notna()
df_dys = df.loc[mask].copy()

print('dyslexic rows', df_dys.shape[0])
print(df_dys['device'].value_counts())

# groups
rv0 = df_dys.loc[df_dys['language']==0, 'wpm'] = wpm[mask & (reader_view==0)]
rv1 = df_dys.loc[df_dys['language']==1, 'wpm'] = wpm[mask & (reader_view==1)]

rv0 = wpm[mask & (reader_view==0)]
rv1 = wpm[mask & (reader_view==1)]

print('n rv0', rv0.shape[0], 'n rv1', rv1.shape[0])

print('mean rv0', rv0.mean(), 'median', rv0.median(), 'std', rv0.std())
print('mean rv1', rv1.mean(), 'median', rv1.median(), 'std', rv1.std())

# t-test (Welch)
t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')
print('t-test', t_stat, p_val)

# Mann-Whitney U
u_stat, p_u = stats.mannwhitneyu(rv1, rv0, alternative='two-sided')
print('mannwhitney', u_stat, p_u)

# effect size (Cohen's d)
# Using pooled SD for d
n1, n0 = rv1.shape[0], rv0.shape[0]
var1, var0 = rv1.var(ddof=1), rv0.var(ddof=1)
pooled = np.sqrt(((n1-1)*var1 + (n0-1)*var0) / (n1+n0-2))
d = (rv1.mean() - rv0.mean()) / pooled
print('cohen_d', d)

# log-transform wpm to reduce skew
log_rv0 = np.log(rv0)
log_rv1 = np.log(rv1)
t_stat_log, p_log = stats.ttest_ind(log_rv1, log_rv0, equal_var=False, nan_policy='omit')
print('log t-test', t_stat_log, p_log)

# effect size on log
var1l, var0l = log_rv1.var(ddof=1), log_rv0.var(ddof=1)
pooled_l = np.sqrt(((n1-1)*var1l + (n0-1)*var0l) / (n1+n0-2))
d_log = (log_rv1.mean() - log_rv0.mean()) / pooled_l
print('cohen_d_log', d_log)
