import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('reading.csv')

# Focus on readers with dyslexia
sub = df[df['dyslexia_bin'] == 1].copy()
sub = sub[(sub['speed'] > 0) & sub['reader_view'].isin([0, 1])]

# Descriptive comparison of raw speed
rv0 = sub[sub['reader_view'] == 0]['speed']
rv1 = sub[sub['reader_view'] == 1]['speed']

raw_means = sub.groupby('reader_view')['speed'].mean()
raw_ttest = stats.ttest_ind(rv1, rv0, equal_var=False)

# Log-speed analysis to reduce skew
sub['log_speed'] = np.log(sub['speed'])
rv0_log = sub[sub['reader_view'] == 0]['log_speed']
rv1_log = sub[sub['reader_view'] == 1]['log_speed']
log_means = sub.groupby('reader_view')['log_speed'].mean()
log_ttest = stats.ttest_ind(rv1_log, rv0_log, equal_var=False)

# Regression with controls (robust SEs)
cols = ['reader_view', 'num_words', 'age', 'retake_trial', 'correct_rate', 'page_id', 'device', 'language']
sub_reg = sub.dropna(subset=cols + ['log_speed'])
formula = 'log_speed ~ reader_view + num_words + age + retake_trial + correct_rate + C(page_id) + C(device) + C(language)'
model = smf.ols(formula, data=sub_reg).fit(cov_type='HC3')

# Print key results
print('Dyslexia subgroup size:', len(sub))
print('Mean speed (reader_view=0):', raw_means.get(0))
print('Mean speed (reader_view=1):', raw_means.get(1))
print('Raw speed t-test:', raw_ttest)
print('Mean log-speed (reader_view=0):', log_means.get(0))
print('Mean log-speed (reader_view=1):', log_means.get(1))
print('Log-speed t-test:', log_ttest)
print('Regression reader_view coef:', model.params.get('reader_view'))
print('Regression reader_view p-value:', model.pvalues.get('reader_view'))
print('Regression reader_view 95% CI:', model.conf_int().loc['reader_view'].tolist())
