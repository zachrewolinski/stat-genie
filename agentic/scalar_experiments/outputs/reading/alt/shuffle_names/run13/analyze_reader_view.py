import pandas as pd
import numpy as np
from scipy import stats

path='reading.csv'
df=pd.read_csv(path)

# Identify key variables
reading_speed = df['running_time']  # derived wpm
reader_view = df['language']        # 1=reader view on, 0=off per metadata

dyslexia = df['dyslexia']

# define dyslexia group: 1 or 2
mask_dys = dyslexia.isin([1.0, 2.0])

# filter to rows with non-missing
sub = df.loc[mask_dys & reader_view.notna() & reading_speed.notna()]

# split
speed_rv = sub.loc[sub['language'] == 1, 'running_time']
speed_no = sub.loc[sub['language'] == 0, 'running_time']

print('dyslexia group size', len(sub), 'rv', len(speed_rv), 'no', len(speed_no))

# summary stats
for label, s in [('ReaderView', speed_rv), ('NoReaderView', speed_no)]:
    print(label, 'mean', s.mean(), 'median', s.median(), 'std', s.std(), 'n', s.shape[0])

# Welch t-test
if len(speed_rv) > 1 and len(speed_no) > 1:
    tstat, pval = stats.ttest_ind(speed_rv, speed_no, equal_var=False, nan_policy='omit')
    print('welch t-test', tstat, pval)

# Mann-Whitney U (nonparam)
if len(speed_rv) > 0 and len(speed_no) > 0:
    ustat, p_u = stats.mannwhitneyu(speed_rv, speed_no, alternative='two-sided')
    print('mannwhitney', ustat, p_u)

# effect size (Cohen d)
mean1, mean0 = speed_rv.mean(), speed_no.mean()
var1, var0 = speed_rv.var(ddof=1), speed_no.var(ddof=1)
# pooled sd for d
n1, n0 = len(speed_rv), len(speed_no)
pooled = np.sqrt(((n1-1)*var1 + (n0-1)*var0)/(n1+n0-2)) if (n1+n0-2)>0 else np.nan
d = (mean1-mean0)/pooled if pooled>0 else np.nan
print('cohen_d', d)

# compare also using log speed to reduce skew
log_speed_rv = np.log1p(speed_rv)
log_speed_no = np.log1p(speed_no)
if len(log_speed_rv)>1 and len(log_speed_no)>1:
    tstat_log, pval_log = stats.ttest_ind(log_speed_rv, log_speed_no, equal_var=False)
    print('welch t-test log', tstat_log, pval_log)
    mean_log_diff = log_speed_rv.mean() - log_speed_no.mean()
    print('mean_log_diff', mean_log_diff)

# simple regression with controls? We'll include word count and device maybe
# Use only rows with needed vars
reg_df = sub[['running_time','language','retake_trial','num_words','correct_rate','education','device','dyslexia','english_native','img_width']].copy()
# convert categorical device/english_native/img_width
# device appears numeric already; english_native and img_width are categorical
reg_df = reg_df.dropna()
# encode english_native and img_width
reg_df = pd.get_dummies(reg_df, columns=['english_native','img_width'], drop_first=True)

# OLS regression
import statsmodels.api as sm
X = reg_df.drop(columns=['running_time'])
X = sm.add_constant(X)
y = reg_df['running_time']
model = sm.OLS(y, X).fit()
print(model.summary().as_text())
