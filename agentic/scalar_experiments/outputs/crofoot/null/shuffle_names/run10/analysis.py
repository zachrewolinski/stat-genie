import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy import stats

# Load data
_df = pd.read_csv('crofoot.csv')

# Outcome: focal group wins (1/0)
win = _df['m_focal']

# According to metadata: f_other = number of individuals in focal group
# win = number of individuals in other group
size_focal = _df['f_other']
size_other = _df['win']
rel_size = size_focal - size_other

# According to metadata: m_other = distance of focal group from its home range center
# n_focal = distance of other group from its home range center
# Contest location advantage captured by relative distance (focal - other)
rel_loc = _df['m_other'] - _df['n_focal']

X = pd.DataFrame({'rel_size': rel_size, 'rel_loc': rel_loc})
X = sm.add_constant(X)

model = sm.Logit(win, X).fit(disp=False)

# Extract stats
params = model.params
conf = model.conf_int()
conf.columns = ['ci_low', 'ci_high']

# Odds ratios and CI
odds = np.exp(params)
odds_ci = np.exp(conf)

# Simple bivariate correlations for sanity (point-biserial)
# (Not primary inference, just descriptive)
rel_size_corr = stats.pointbiserialr(win, rel_size)
rel_loc_corr = stats.pointbiserialr(win, rel_loc)

# Summarize
print('Logit coefficients:')
print(params)
print('\nP-values:')
print(model.pvalues)
print('\n95% CI:')
print(conf)
print('\nOdds ratios:')
print(odds)
print('\nOdds ratio 95% CI:')
print(odds_ci)
print('\nPseudo R2:', model.prsquared)
print('\nPoint-biserial correlations:')
print('rel_size:', rel_size_corr)
print('rel_loc:', rel_loc_corr)
