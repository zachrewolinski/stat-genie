import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind

# Load data

df = pd.read_csv('reading.csv')

# Reconstruct dyslexia categories by rounding to the nearest integer
# (metadata says 0=no dyslexia, 1=dyslexia, 2=severe dyslexia)
# Values appear noisy, so we round and clamp.
df['dyslexia_round'] = np.clip(np.rint(df['dyslexia']), 0, 2).astype(int)

dyslexic = df[df['dyslexia_round'] >= 1].copy()

# Derived reading speed (words per minute) from adjusted running time
# adjusted_running_time is in ms per metadata

dyslexic['wpm'] = dyslexic['num_words'] / dyslexic['adjusted_running_time'] * 60000

dyslexic['log_speed'] = np.log(dyslexic['speed'])

dyslexic['log_wpm'] = np.log(dyslexic['wpm'])

# Group summaries

def group_stats(series):
    return pd.Series({
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'n': series.shape[0]
    })

speed_stats = dyslexic.groupby('reader_view')['speed'].apply(group_stats)
wpm_stats = dyslexic.groupby('reader_view')['wpm'].apply(group_stats)

# T-tests (Welch) on log-transformed outcomes
rv1_speed = dyslexic[dyslexic['reader_view'] == 1]['log_speed']
rv0_speed = dyslexic[dyslexic['reader_view'] == 0]['log_speed']

rv1_wpm = dyslexic[dyslexic['reader_view'] == 1]['log_wpm']
rv0_wpm = dyslexic[dyslexic['reader_view'] == 0]['log_wpm']

speed_t = ttest_ind(rv1_speed, rv0_speed, usevar='unequal')
wpm_t = ttest_ind(rv1_wpm, rv0_wpm, usevar='unequal')

# Regression with basic controls
# Use log_speed to reduce skew
model = smf.ols(
    'log_speed ~ reader_view + num_words + C(page_id) + C(device) + age + retake_trial',
    data=dyslexic
).fit()

# Output key results
print('Dyslexic sample size:', len(dyslexic))
print('\nSpeed stats (raw speed):')
print(speed_stats)
print('\nWPM stats (derived):')
print(wpm_stats)

print('\nWelch t-test on log(speed): statistic=%.4f, p=%.4f' % (speed_t[0], speed_t[1]))
print('Welch t-test on log(wpm): statistic=%.4f, p=%.4f' % (wpm_t[0], wpm_t[1]))

print('\nRegression coefficient for reader_view (log_speed):')
print('coef=%.6f, p=%.4f' % (model.params['reader_view'], model.pvalues['reader_view']))
