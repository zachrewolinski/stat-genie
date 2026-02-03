import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
DF_PATH = 'reading.csv'
df = pd.read_csv(DF_PATH)

# Focus on participants with dyslexia
# dyslexia_bin: 1 indicates dyslexia
subset = df[df['dyslexia_bin'] == 1].copy()

# Basic group summaries
rv0 = subset[subset['reader_view'] == 0]['speed'].dropna()
rv1 = subset[subset['reader_view'] == 1]['speed'].dropna()

print('Dyslexia subset rows:', subset.shape[0])
print('Reader view OFF count:', rv0.shape[0])
print('Reader view ON  count:', rv1.shape[0])
print('Mean speed OFF:', rv0.mean())
print('Mean speed ON :', rv1.mean())
print('Median speed OFF:', rv0.median())
print('Median speed ON :', rv1.median())

# Compare log-speed to reduce skew
log_rv0 = np.log(rv0)
log_rv1 = np.log(rv1)

ttest = stats.ttest_ind(log_rv1, log_rv0, equal_var=False, nan_policy='omit')
print('\nWelch t-test on log(speed):')
print(ttest)

# Regression with controls, clustered SEs by participant
cols = [
    'reader_view', 'num_words', 'page_id', 'device', 'age', 'gender',
    'education', 'language', 'english_native', 'retake_trial',
    'correct_rate', 'Flesch_Kincaid', 'img_width', 'uuid', 'speed'
]
reg_df = subset[cols].dropna().copy()
reg_df['log_speed'] = np.log(reg_df['speed'])

for c in ['page_id', 'device', 'education', 'language', 'english_native']:
    reg_df[c] = reg_df[c].astype('category')

formula = (
    'log_speed ~ reader_view + num_words + C(page_id) + C(device) + age + '
    'C(gender) + C(education) + C(language) + C(english_native) + '
    'retake_trial + correct_rate + Flesch_Kincaid + img_width'
)
model = smf.ols(formula, data=reg_df).fit(
    cov_type='cluster', cov_kwds={'groups': reg_df['uuid']}
)

coef = model.params['reader_view']
se = model.bse['reader_view']
ci_low, ci_high = coef - 1.96 * se, coef + 1.96 * se
pct = (np.exp(coef) - 1) * 100
pct_low, pct_high = (np.exp(ci_low) - 1) * 100, (np.exp(ci_high) - 1) * 100

print('\nRegression (log-speed, clustered by uuid):')
print('reader_view coef:', coef)
print('reader_view SE:', se)
print('95% CI:', (ci_low, ci_high))
print('Approx % change:', pct)
print('Approx % change 95% CI:', (pct_low, pct_high))
print('p-value:', model.pvalues['reader_view'])
