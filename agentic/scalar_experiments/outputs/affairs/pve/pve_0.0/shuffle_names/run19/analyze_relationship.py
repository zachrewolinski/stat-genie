import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data

df = pd.read_csv('affairs.csv')

# Identify columns
children_col = 'religiousness'  # yes/no, per metadata description
affairs_col = 'age'             # per metadata description (affairs frequency)

# Encode children as binary
children_bin = df[children_col].map({'no': 0, 'yes': 1})

affairs = df[affairs_col]

# Group stats
grp = df.groupby(children_col)[affairs_col].agg(['mean','median','std','count'])

# t-test (Welch)
no = df[df[children_col] == 'no'][affairs_col]
yes = df[df[children_col] == 'yes'][affairs_col]

welch = stats.ttest_ind(yes, no, equal_var=False)

# Mann-Whitney U (two-sided)
try:
    mw = stats.mannwhitneyu(yes, no, alternative='two-sided')
except Exception as e:
    mw = None

# Cohen's d (using pooled std)
pooled_std = np.sqrt(((len(yes)-1)*yes.var(ddof=1) + (len(no)-1)*no.var(ddof=1)) / (len(yes)+len(no)-2))
cohen_d = (yes.mean() - no.mean()) / pooled_std if pooled_std > 0 else np.nan

# Simple linear regression: affairs ~ children
X = sm.add_constant(children_bin)
model = sm.OLS(affairs, X).fit()

print('Group stats:\n', grp)
print('\nWelch t-test:', welch)
if mw:
    print('Mann-Whitney U:', mw)
print('\nCohen d:', cohen_d)
print('\nOLS summary (affairs ~ children):')
print(model.summary())
