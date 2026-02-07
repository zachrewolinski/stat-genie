import pandas as pd
import numpy as np
from scipy import stats

# Load data

df = pd.read_csv('affairs.csv')

# Map columns based on metadata shuffle
# affairs frequency is in column 'age'
# children indicator is in column 'religiousness' (yes/no)

# prepare

# convert

affairs = df['age']
children = df['religiousness']

# group stats

grouped = df.groupby(children)['age'].agg(['count','mean','std'])
print(grouped)

# difference

mean_yes = grouped.loc['yes','mean']
mean_no = grouped.loc['no','mean']

diff = mean_yes - mean_no
print('mean_yes', mean_yes, 'mean_no', mean_no, 'diff yes-no', diff)

# t-test

yes_vals = df.loc[children=='yes','age']
no_vals = df.loc[children=='no','age']

t_stat, p_val = stats.ttest_ind(yes_vals, no_vals, equal_var=False)
print('t', t_stat, 'p', p_val)

# effect size (Cohen d) using pooled SD

n1 = len(yes_vals)
n2 = len(no_vals)

s1 = yes_vals.std(ddof=1)
s2 = no_vals.std(ddof=1)
sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))

d = (mean_yes - mean_no)/sp
print('cohen d', d)

# also compute proportion with any affairs (>0)

prop_yes = (yes_vals>0).mean()
prop_no = (no_vals>0).mean()
print('prop any yes', prop_yes, 'prop any no', prop_no, 'diff', prop_yes-prop_no)

# chi-square test for any affairs

cont = pd.crosstab(children, df['age']>0)
chi2, p_chi, _, _ = stats.chi2_contingency(cont)
print(cont)
print('chi2', chi2, 'p', p_chi)
