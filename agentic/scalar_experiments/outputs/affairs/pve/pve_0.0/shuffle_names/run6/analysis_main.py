import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('affairs.csv')

# Identify columns based on metadata descriptions (names are shuffled)
children_col = 'religiousness'  # yes/no
affairs_col = 'age'             # described as affairs frequency

# Map children yes/no to binary
children = df[children_col].map({'yes': 1, 'no': 0})

# Outcome
y = df[affairs_col]

# Group stats
stats_by_group = df.groupby(children_col)[affairs_col].agg(['count','mean','std'])
print('Group stats (affairs by children):')
print(stats_by_group)

# Welch t-test
no_group = y[children == 0]
yes_group = y[children == 1]

t_stat, p_value = stats.ttest_ind(no_group, yes_group, equal_var=False, nan_policy='omit')

# Cohen's d (using pooled SD)
mean_diff = no_group.mean() - yes_group.mean()
pooled_sd = np.sqrt(((no_group.std(ddof=1) ** 2) + (yes_group.std(ddof=1) ** 2)) / 2)
cohens_d = mean_diff / pooled_sd if pooled_sd != 0 else np.nan

# OLS regression for robustness
X = sm.add_constant(children)
model = sm.OLS(y, X).fit()

print('\nWelch t-test:')
print('t-stat', t_stat, 'p-value', p_value)
print('mean(no children) =', no_group.mean(), 'mean(children) =', yes_group.mean())
print('mean diff (no - yes) =', mean_diff)
print("Cohen's d =", cohens_d)

print('\nOLS:')
print(model.summary().tables[1])
