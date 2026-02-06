import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('reading.csv')

# Focus on participants with dyslexia
sub = df[df['dyslexia_bin'] == 1].copy()

# Basic cleaning
sub = sub[sub['speed'] > 0].copy()

# Group summaries
summary = sub.groupby('reader_view')['speed'].agg(['count', 'mean', 'median']).rename(index={0: 'No reader view', 1: 'Reader view'})

# Welch t-test on raw speed
s0 = sub.loc[sub['reader_view'] == 0, 'speed']
s1 = sub.loc[sub['reader_view'] == 1, 'speed']
raw_ttest = stats.ttest_ind(s1, s0, equal_var=False)

# Log-speed analysis to reduce skew
sub['log_speed'] = np.log(sub['speed'])
ls0 = sub.loc[sub['reader_view'] == 0, 'log_speed']
ls1 = sub.loc[sub['reader_view'] == 1, 'log_speed']
log_ttest = stats.ttest_ind(ls1, ls0, equal_var=False)

# Regression with controls
formula = (
    'log_speed ~ reader_view + num_words + Flesch_Kincaid + C(page_id) + '
    'C(device) + C(language) + age + C(gender) + C(education) + '
    'C(english_native) + retake_trial'
)
model = smf.ols(formula, data=sub).fit(cov_type='HC3')

# Output results
print('Dyslexia subset summary (speed):')
print(summary)
print('\nWelch t-test (speed):', raw_ttest)
print('Welch t-test (log speed):', log_ttest)
print('\nRegression (log speed) coefficient for reader_view:')
print(model.params['reader_view'], 'p=', model.pvalues['reader_view'])
