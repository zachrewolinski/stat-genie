import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('affairs.csv')
# map columns
# children indicator is in 'religiousness' column (yes/no)
children = df['religiousness'].map({'yes': 1, 'no': 0})
# affairs count is in 'age' column
affairs = df['age']

df2 = pd.DataFrame({'children': children, 'affairs': affairs})

summary = df2.groupby('children').agg(
    n=('affairs', 'size'),
    mean=('affairs', 'mean'),
    median=('affairs', 'median'),
    std=('affairs', 'std'),
    prop_any=('affairs', lambda x: (x > 0).mean()),
)
print('Summary by children (0=no,1=yes):')
print(summary)

# difference in means
mean_no = summary.loc[0, 'mean']
mean_yes = summary.loc[1, 'mean']
mean_diff = mean_yes - mean_no
print('Mean difference (yes - no):', mean_diff)

# Cohen's d
s1 = df2.loc[df2['children'] == 1, 'affairs']
s0 = df2.loc[df2['children'] == 0, 'affairs']
# pooled std
n1, n0 = len(s1), len(s0)
var1, var0 = s1.var(ddof=1), s0.var(ddof=1)
sp = np.sqrt(((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2))
cohen_d = (s1.mean() - s0.mean()) / sp
print('Cohen d:', cohen_d)

# t-test
print('t-test:')
print(stats.ttest_ind(s1, s0, equal_var=True))

# Mann-Whitney U
print('Mann-Whitney U:')
print(stats.mannwhitneyu(s1, s0, alternative='two-sided'))

# proportion any affairs chi-square
cont = pd.crosstab(df2['children'], df2['affairs'] > 0)
print('Contingency (children x any_affairs):')
print(cont)
chi2, p, dof, exp = stats.chi2_contingency(cont)
print('Chi-square p:', p)
