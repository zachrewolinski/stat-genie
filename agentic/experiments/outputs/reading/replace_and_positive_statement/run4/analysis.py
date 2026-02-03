import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = 'reading.csv'
df = pd.read_csv(path)

# Use dyslexia_bin as continuous severity; define dyslexia group as >= median
median_dys = df['dyslexia_bin'].median()
dys = df[df['dyslexia_bin'] >= median_dys].copy()
non = df[df['dyslexia_bin'] < median_dys].copy()

# Summary stats for dyslexia group
mean_speed_rv1 = dys[dys['reader_view'] == 1]['speed'].mean()
mean_speed_rv0 = dys[dys['reader_view'] == 0]['speed'].mean()
mean_diff = mean_speed_rv1 - mean_speed_rv0

# Two-sample t-test within dyslexia group
rv1 = dys[dys['reader_view'] == 1]['speed']
rv0 = dys[dys['reader_view'] == 0]['speed']
t_stat, p_val = stats.ttest_ind(rv1, rv0, equal_var=False)

# Regression with interaction and controls
formula = (
    'speed ~ reader_view * dyslexia_bin + num_words + Flesch_Kincaid + '
    'age + correct_rate + C(device) + C(page_id)'
)
model = smf.ols(formula, data=df).fit()

# Marginal effect of reader_view at different dyslexia levels
levels = [0.25, 0.5, 0.75, 1.0]
interaction = model.params['reader_view:dyslexia_bin']
base = model.params['reader_view']
marginal_effects = {lvl: base + interaction * lvl for lvl in levels}

# Print key results
print('Median dyslexia_bin:', median_dys)
print('Dyslexia group (>= median) mean speed, reader_view=1:', mean_speed_rv1)
print('Dyslexia group (>= median) mean speed, reader_view=0:', mean_speed_rv0)
print('Mean difference (rv1 - rv0):', mean_diff)
print('T-test t-stat:', t_stat, 'p-value:', p_val)
print('\nRegression summary (selected):')
print('reader_view coef:', model.params['reader_view'])
print('reader_view:dyslexia_bin coef:', interaction, 'p-value:', model.pvalues['reader_view:dyslexia_bin'])
print('Marginal effects of reader_view at dyslexia_bin levels:', marginal_effects)
