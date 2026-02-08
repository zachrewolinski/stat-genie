import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('affairs.csv')

# Ensure expected categories
# children: yes/no

# Any affair indicator
df['any_affair'] = df['affairs'] > 0

summary = df.groupby('children').agg(
    n=('affairs', 'size'),
    mean_affairs=('affairs', 'mean'),
    median_affairs=('affairs', 'median'),
    prop_any=('any_affair', 'mean')
)

# Differences
mean_diff = summary.loc['yes', 'mean_affairs'] - summary.loc['no', 'mean_affairs']
prop_diff = summary.loc['yes', 'prop_any'] - summary.loc['no', 'prop_any']

# t-test on affairs
yes_affairs = df.loc[df['children'] == 'yes', 'affairs']
no_affairs = df.loc[df['children'] == 'no', 'affairs']

t_stat, t_p = stats.ttest_ind(yes_affairs, no_affairs, equal_var=False)

# Mann-Whitney U (non-parametric)
# Use two-sided for general difference
u_stat, u_p = stats.mannwhitneyu(yes_affairs, no_affairs, alternative='two-sided')

# Proportion test for any affair
# Use two-proportion z-test
n1 = summary.loc['yes', 'n']
n0 = summary.loc['no', 'n']
x1 = df.loc[df['children']=='yes', 'any_affair'].sum()
x0 = df.loc[df['children']=='no', 'any_affair'].sum()

p1 = x1 / n1
p0 = x0 / n0

p_pool = (x1 + x0) / (n1 + n0)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n0))
if se == 0:
    z = np.nan
    p_prop = np.nan
else:
    z = (p1 - p0) / se
    p_prop = 2 * (1 - stats.norm.cdf(abs(z)))

# Effect size for difference in means (Cohen's d)
# Use pooled std for d
s1 = yes_affairs.std(ddof=1)
s0 = no_affairs.std(ddof=1)
sp = np.sqrt(((n1 - 1)*s1**2 + (n0 - 1)*s0**2) / (n1 + n0 - 2))
cohen_d = (yes_affairs.mean() - no_affairs.mean()) / sp if sp != 0 else np.nan

print('Summary by children:')
print(summary)
print('\nMean difference (yes - no):', mean_diff)
print('Prop any difference (yes - no):', prop_diff)
print('\nT-test (affairs): t=', t_stat, 'p=', t_p)
print('Mann-Whitney U: U=', u_stat, 'p=', u_p)
print('Two-proportion z-test (any affair): z=', z, 'p=', p_prop)
print('Cohen d (means):', cohen_d)
