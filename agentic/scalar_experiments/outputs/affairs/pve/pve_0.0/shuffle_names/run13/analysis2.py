import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('affairs.csv')

# Mapping based on metadata descriptions
children_col = 'religiousness'  # yes/no
affairs_col = 'age'  # described as affairs frequency

# Drop missing
data = df[[children_col, affairs_col]].dropna().copy()

# Encode children: yes=1, no=0
children_map = {'yes': 1, 'no': 0}
if set(data[children_col].unique()) <= set(children_map.keys()):
    data['children'] = data[children_col].map(children_map)
else:
    # fallback for unexpected coding
    data['children'] = data[children_col].astype('category').cat.codes

# Group stats
summary = data.groupby('children')[affairs_col].agg(['count', 'mean', 'std'])
print('Group summary (children=0 no, 1 yes):')
print(summary)

# T-test
no_vals = data.loc[data['children'] == 0, affairs_col]
yes_vals = data.loc[data['children'] == 1, affairs_col]

# Welch's t-test
if len(no_vals) > 1 and len(yes_vals) > 1:
    t_stat, p_val = stats.ttest_ind(yes_vals, no_vals, equal_var=False)
    print('\nWelch t-test (yes - no):')
    print('t=', t_stat, 'p=', p_val)

    # Cohen's d (Hedges g) for unequal sizes
    n1, n0 = len(yes_vals), len(no_vals)
    s1, s0 = yes_vals.std(ddof=1), no_vals.std(ddof=1)
    # pooled SD
    s_pooled = np.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2) / (n1 + n0 - 2))
    d = (yes_vals.mean() - no_vals.mean()) / s_pooled if s_pooled > 0 else np.nan
    # Hedges g
    correction = 1 - (3 / (4*(n1+n0) - 9))
    g = d * correction if np.isfinite(d) else np.nan
    print('Cohen d=', d, 'Hedges g=', g)

# Mann-Whitney U test (non-parametric)
if len(no_vals) > 0 and len(yes_vals) > 0:
    u_stat, p_u = stats.mannwhitneyu(yes_vals, no_vals, alternative='two-sided')
    print('\nMann-Whitney U:')
    print('U=', u_stat, 'p=', p_u)

# OLS regression: affairs ~ children
X = sm.add_constant(data['children'])
model = sm.OLS(data[affairs_col], X).fit(cov_type='HC3')
print('\nOLS regression (affairs ~ children):')
print(model.summary())
