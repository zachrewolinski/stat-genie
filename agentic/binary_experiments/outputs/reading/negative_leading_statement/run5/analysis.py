import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('reading.csv')

# Focus on participants with dyslexia
dys = df[df['dyslexia_bin'] == 1].copy()

# Basic group stats
group_stats = dys.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std'])

# Welch t-test on speed
rv1 = dys[dys['reader_view'] == 1]['speed']
rv0 = dys[dys['reader_view'] == 0]['speed']
t_res = stats.ttest_ind(rv1, rv0, equal_var=False, nan_policy='omit')

# Also examine log-speed to reduce skew
dys['log_speed'] = np.log(dys['speed'])
rv1_log = dys[dys['reader_view'] == 1]['log_speed']
rv0_log = dys[dys['reader_view'] == 0]['log_speed']
t_res_log = stats.ttest_ind(rv1_log, rv0_log, equal_var=False, nan_policy='omit')

# Regression controlling for page and text length
# Use categorical for page_id and device; include num_words
reg_df = dys.dropna(subset=['log_speed', 'reader_view', 'num_words', 'page_id', 'device'])
model = smf.ols('log_speed ~ reader_view + num_words + C(page_id) + C(device)', data=reg_df).fit()

# Print results
print('Group stats (speed) for dyslexia_bin=1 by reader_view')
print(group_stats)
print('\nWelch t-test speed: t=%.4f, p=%.6f' % (t_res.statistic, t_res.pvalue))
print('Welch t-test log_speed: t=%.4f, p=%.6f' % (t_res_log.statistic, t_res_log.pvalue))
print('\nRegression on log_speed with controls (reader_view coef)')
print('coef=%.6f, p=%.6f' % (model.params['reader_view'], model.pvalues['reader_view']))
print(model.summary().tables[1])
