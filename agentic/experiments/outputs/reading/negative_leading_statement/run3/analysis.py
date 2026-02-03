import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
csv_path = 'reading.csv'
df = pd.read_csv(csv_path)

# Focus on participants with dyslexia
sub = df[df['dyslexia_bin'] == 1].copy()

# Descriptive stats by reader view
summary = sub.groupby('reader_view')['speed'].agg(['count', 'mean', 'median', 'std'])
print('Speed summary for dyslexia_bin=1 by reader_view')
print(summary)
print()

# Welch t-test on raw speed
s0 = sub[sub['reader_view'] == 0]['speed']
s1 = sub[sub['reader_view'] == 1]['speed']

welch_raw = stats.ttest_ind(s1, s0, equal_var=False, nan_policy='omit')
print('Welch t-test (raw speed):', welch_raw)
print('Mean difference (reader_view=1 minus 0):', s1.mean() - s0.mean())
print()

# Welch t-test on log speed to reduce skew
sub = sub[(sub['speed'] > 0)].copy()
sub['log_speed'] = np.log(sub['speed'])

s0l = sub[sub['reader_view'] == 0]['log_speed']
s1l = sub[sub['reader_view'] == 1]['log_speed']
welch_log = stats.ttest_ind(s1l, s0l, equal_var=False, nan_policy='omit')
print('Welch t-test (log speed):', welch_log)
print()

# Regression controlling for text/page characteristics and demographics
# Use robust SE to reduce sensitivity to heteroskedasticity
model = smf.ols(
    'log_speed ~ reader_view + num_words + Flesch_Kincaid + C(page_id) + C(device) + age + C(english_native)',
    data=sub
).fit(cov_type='HC3')

print('Regression (log_speed) with controls:')
print(model.summary().tables[1])
