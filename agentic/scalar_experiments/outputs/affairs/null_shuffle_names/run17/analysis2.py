import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('affairs.csv')
# Map children indicator from column 'religiousness'
child_col = 'religiousness'
# affair engagement variable from column 'age'
affair_col = 'age'

# Encode children yes/no
has_children = df[child_col].map({'yes': 1, 'no': 0})

# Compute basic stats
groups = {}
for val, label in [(1, 'yes'), (0, 'no')]:
    sub = df[has_children == val][affair_col]
    groups[label] = sub
    print(label, 'n', len(sub), 'mean', sub.mean(), 'median', sub.median(), 'prop_any', (sub > 0).mean())

# Difference in means
mean_diff = groups['no'].mean() - groups['yes'].mean()  # positive means children lower affairs
print('Mean diff (no - yes):', mean_diff)

# t-test
res = stats.ttest_ind(groups['no'], groups['yes'], equal_var=False)
print('t-test:', res)

# Mann-Whitney
u = stats.mannwhitneyu(groups['no'], groups['yes'], alternative='two-sided')
print('mannwhitney:', u)

# Effect size (Cohen d)
mean1 = groups['no'].mean()
mean2 = groups['yes'].mean()
std1 = groups['no'].std(ddof=1)
std2 = groups['yes'].std(ddof=1)
# pooled std
n1 = len(groups['no'])
n2 = len(groups['yes'])
pooled = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
cohen_d = (mean1 - mean2) / pooled
print('Cohen d (no - yes):', cohen_d)

# Difference in proportion any affair
p_no = (groups['no'] > 0).mean()
p_yes = (groups['yes'] > 0).mean()
print('prop any diff (no - yes):', p_no - p_yes)

# two-proportion z-test
count = np.array([(groups['no'] > 0).sum(), (groups['yes'] > 0).sum()])
obs = np.array([n1, n2])
# Manual z-test
p_pool = count.sum() / obs.sum()
se = np.sqrt(p_pool * (1 - p_pool) * (1/obs[0] + 1/obs[1]))
z = (p_no - p_yes) / se
pval = 2 * (1 - stats.norm.cdf(abs(z)))
print('z-test proportion: z', z, 'p', pval)

