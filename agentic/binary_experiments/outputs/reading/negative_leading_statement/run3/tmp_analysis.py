import pandas as pd
import numpy as np
from scipy import stats

csv_path = 'reading.csv'
df = pd.read_csv(csv_path)

# focus on dyslexia
sub = df[df['dyslexia_bin'] == 1]

# basic stats
for rv in [0,1]:
    s = sub[sub['reader_view'] == rv]['speed']
    print('reader_view', rv, 'n', len(s), 'mean', s.mean(), 'median', s.median())

# t-test Welch
s0 = sub[sub['reader_view']==0]['speed']
s1 = sub[sub['reader_view']==1]['speed']

print('welch t', stats.ttest_ind(s1, s0, equal_var=False, nan_policy='omit'))

# log transform
s0l = np.log(s0)
s1l = np.log(s1)
print('welch t log', stats.ttest_ind(s1l, s0l, equal_var=False, nan_policy='omit'))

# effect size (Cohen d)
mean_diff = s1.mean() - s0.mean()
print('mean diff', mean_diff)

# regression with controls
import statsmodels.formula.api as smf

sub2 = sub.copy()
sub2['log_speed'] = np.log(sub2['speed'])
model = smf.ols('log_speed ~ reader_view + num_words + Flesch_Kincaid + C(page_id) + C(device) + age + C(english_native)', data=sub2).fit(cov_type='HC3')
print(model.summary().tables[1])
