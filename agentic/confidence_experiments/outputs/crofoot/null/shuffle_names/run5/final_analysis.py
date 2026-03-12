import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
_df = pd.read_csv('crofoot.csv')

# Column mapping based on value ranges and metadata
outcome = 'm_focal'  # binary win indicator
focal_size = 'f_other'  # total individuals in focal group
other_size = 'win'      # total individuals in other group

dist_focal = 'm_other'  # distance from focal home-range center
# distance from other home-range center
# (using the remaining large-range column)
dist_other = 'n_focal'

# Derived variables
_df['relative_size'] = _df[focal_size] - _df[other_size]
_df['relative_size_log'] = np.log(_df[focal_size] / _df[other_size])
_df['relative_location'] = _df[dist_focal] - _df[dist_other]

# Logistic regression with size difference
X = sm.add_constant(_df[['relative_size', 'relative_location']])
model = sm.Logit(_df[outcome], X)
res = model.fit(disp=False)

# Logistic regression with size log-ratio
X2 = sm.add_constant(_df[['relative_size_log', 'relative_location']])
model2 = sm.Logit(_df[outcome], X2)
res2 = model2.fit(disp=False)

# Simple win-rate comparisons
_df['rel_size_sign'] = np.sign(_df['relative_size'])
_df['rel_loc_sign'] = np.sign(_df['relative_location'])

win_by_size = _df.groupby('rel_size_sign')[outcome].mean()
win_by_loc = _df.groupby('rel_loc_sign')[outcome].mean()

# Two-sample t-test for win rates by size sign (larger vs smaller)
larger = _df[_df['relative_size'] > 0][outcome]
smaller = _df[_df['relative_size'] < 0][outcome]
if len(larger) > 1 and len(smaller) > 1:
    t_stat, t_p = stats.ttest_ind(larger, smaller, equal_var=False)
else:
    t_stat, t_p = np.nan, np.nan

# Two-sample t-test for win rates by location sign (closer to focal vs closer to other)
closer_focal = _df[_df['relative_location'] < 0][outcome]
closer_other = _df[_df['relative_location'] > 0][outcome]
if len(closer_focal) > 1 and len(closer_other) > 1:
    t_stat_loc, t_p_loc = stats.ttest_ind(closer_focal, closer_other, equal_var=False)
else:
    t_stat_loc, t_p_loc = np.nan, np.nan

print('N:', len(_df))
print('\nLogit (relative_size + relative_location):')
print(res.summary())
print('\nLogit (relative_size_log + relative_location):')
print(res2.summary())

print('\nWin rate by relative_size sign (negative=smaller, positive=larger):')
print(win_by_size)
print('t-test p-value:', t_p)

print('\nWin rate by relative_location sign (negative=closer to focal):')
print(win_by_loc)
print('t-test p-value:', t_p_loc)

# Effect sizes (odds ratios)
print('\nOdds ratios (diff model):')
print(np.exp(res.params))
print('\nOdds ratios (log model):')
print(np.exp(res2.params))
