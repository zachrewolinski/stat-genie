import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind

# Load data
DATA_PATH = 'reading.csv'

df = pd.read_csv(DATA_PATH)

# Focus on higher dyslexia severity group (top quartile of dyslexia_bin)
thr = df['dyslexia_bin'].quantile(0.75)
high = df[df['dyslexia_bin'] >= thr]

rv1 = high[high['reader_view'] == 1]['speed']
rv0 = high[high['reader_view'] == 0]['speed']

# Welch t-test for difference in mean speed
if len(rv1) > 1 and len(rv0) > 1:
    t_stat, p_val, dof = ttest_ind(rv1, rv0, usevar='unequal')
else:
    t_stat, p_val, dof = np.nan, np.nan, np.nan

mean_diff = rv1.mean() - rv0.mean()

# Regression with interaction to estimate effect across dyslexia severity
X = df[['reader_view', 'dyslexia_bin']].copy()
X['interaction'] = X['reader_view'] * X['dyslexia_bin']
X = sm.add_constant(X)
model = sm.OLS(df['speed'], X).fit()

# Effect of reader_view at high dyslexia threshold
b1 = model.params['reader_view']
b3 = model.params['interaction']
# Use delta method for SE
cov = model.cov_params()
se_eff = np.sqrt(
    cov.loc['reader_view', 'reader_view'] +
    (thr ** 2) * cov.loc['interaction', 'interaction'] +
    2 * thr * cov.loc['reader_view', 'interaction']
)

print('High dyslexia threshold (75th percentile):', thr)
print('High dyslexia sample size:', len(high))
print('Reader view=1 mean speed:', rv1.mean())
print('Reader view=0 mean speed:', rv0.mean())
print('Mean difference (rv1 - rv0):', mean_diff)
print('Welch t-test: t=%.3f, p=%.4f, dof=%.1f' % (t_stat, p_val, dof))

print('\nRegression (speed ~ reader_view + dyslexia_bin + interaction)')
print(model.summary().tables[1])

print('\nEffect of reader_view at high dyslexia threshold:')
print('Effect (b1 + b3*thr) =', b1 + b3 * thr)
print('SE (delta method) =', se_eff)
