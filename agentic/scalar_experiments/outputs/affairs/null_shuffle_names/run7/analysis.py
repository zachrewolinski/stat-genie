import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('affairs.csv')

# Identify columns
# Based on metadata mismatch, infer: 'age' looks like affairs frequency, 'religiousness' looks like children (yes/no).

outcome = 'age'
children_col = 'religiousness'

# Basic checks
print('Unique outcome values:', sorted(df[outcome].unique()))
print('Children counts:', df[children_col].value_counts())

# Map children yes/no
# Assume 'yes' means has children

# Compute mean affair frequency by children
means = df.groupby(children_col)[outcome].mean()
stds = df.groupby(children_col)[outcome].std()
counts = df.groupby(children_col)[outcome].count()
print('Means:', means)
print('Stds:', stds)
print('Counts:', counts)

# Difference in means
if 'yes' in means and 'no' in means:
    diff = means['yes'] - means['no']
    print('Mean diff (yes - no):', diff)

# Effect size (Cohen d) using pooled std
    n1, n2 = counts['yes'], counts['no']
    s1, s2 = stds['yes'], stds['no']
    pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/(n1+n2-2))
    d = diff / pooled if pooled != 0 else np.nan
    print('Cohen d:', d)

# t-test
    tstat, pval = stats.ttest_ind(
        df[df[children_col]=='yes'][outcome],
        df[df[children_col]=='no'][outcome],
        equal_var=False,
        nan_policy='omit'
    )
    print('t-test p:', pval)

# Any-affair indicator
any_affair = (df[outcome] > 0).astype(int)
prop = df.groupby(children_col)[outcome].apply(lambda s: (s>0).mean())
print('Any affair proportions:', prop)
if 'yes' in prop and 'no' in prop:
    print('Prop diff (yes - no):', prop['yes'] - prop['no'])

# Chi-square for any-affair
if 'yes' in df[children_col].unique() and 'no' in df[children_col].unique():
    table = pd.crosstab(df[children_col], any_affair)
    chi2, pchi, dof, exp = stats.chi2_contingency(table)
    print('Chi-square p:', pchi)
    print('Crosstab:\n', table)
