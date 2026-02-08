import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('affairs.csv')

# Map columns based on metadata mismatch
# 'age' column seems to be affairs frequency categories
# 'religiousness' column is children yes/no

affairs = df['age']
children = df['religiousness']

# binary any affair >0
any_affair = (affairs > 0).astype(int)

summary = df.groupby(children)['age'].agg(['count','mean','median','std'])
print('Affairs (age column) summary by children:')
print(summary)

# proportions with any affair
prop = df.groupby(children)[any_affair.name].mean()
print('\nProportion with any affair by children:')
print(prop)

# effect sizes
mean_yes = df.loc[children=='yes','age'].mean()
mean_no = df.loc[children=='no','age'].mean()
print('\nMean difference (yes - no):', mean_yes - mean_no)

# Mann-Whitney U test (non-parametric)
try:
    u, p = stats.mannwhitneyu(df.loc[children=='yes','age'], df.loc[children=='no','age'], alternative='two-sided')
    print('Mann-Whitney U p:', p)
except Exception as e:
    print('MWU error', e)

# t-test
try:
    t, p2 = stats.ttest_ind(df.loc[children=='yes','age'], df.loc[children=='no','age'], equal_var=False)
    print('Welch t-test p:', p2)
except Exception as e:
    print('t-test error', e)

# effect size Cohen's d
x = df.loc[children=='yes','age']
y = df.loc[children=='no','age']

nx, ny = len(x), len(y)
varx, vary = x.var(ddof=1), y.var(ddof=1)
sp = ((nx-1)*varx + (ny-1)*vary) / (nx+ny-2)
d = (x.mean()-y.mean())/np.sqrt(sp)
print('Cohen d:', d)

# Differences in any affair proportions
p_yes = any_affair[children=='yes'].mean()
p_no = any_affair[children=='no'].mean()
print('Any affair diff (yes-no):', p_yes - p_no)

# chi-square test for any affair
cont = pd.crosstab(children, any_affair)
print('\nContingency table children vs any_affair')
print(cont)
chi2, pchi, dof, exp = stats.chi2_contingency(cont)
print('Chi-square p:', pchi)
