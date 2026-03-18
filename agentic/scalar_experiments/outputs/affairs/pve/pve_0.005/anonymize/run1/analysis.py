import pandas as pd
import json
from scipy import stats
import numpy as np

df = pd.read_csv('affairs.csv')

# identify columns
# feature6 children yes/no, feature2 affairs

children = df['feature6']
affairs = df['feature2']

# summary stats by children
summary = df.groupby('feature6')['feature2'].agg(['count','mean','median','std'])

# t-test (Welch)
# drop missing
mask_yes = children == 'yes'
mask_no = children == 'no'

x = affairs[mask_yes].dropna()
y = affairs[mask_no].dropna()

t_stat, p_val = stats.ttest_ind(x, y, equal_var=False)

# effect size: Cohen's d (Welch)
# pooled SD for unequal n
nx, ny = len(x), len(y)
var_x, var_y = x.var(ddof=1), y.var(ddof=1)
# use weighted pooled SD
pooled_sd = np.sqrt(((nx-1)*var_x + (ny-1)*var_y) / (nx+ny-2))
cohen_d = (x.mean() - y.mean()) / pooled_sd if pooled_sd != 0 else np.nan

# Mann-Whitney U (non-param) two-sided
u_stat, u_p = stats.mannwhitneyu(x, y, alternative='two-sided')

# difference in means and CI via bootstrap
rng = np.random.default_rng(0)

def bootstrap_diff(x, y, n_boot=5000):
    diffs = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs.append(xb.mean() - yb.mean())
    diffs = np.array(diffs)
    return np.percentile(diffs, [2.5, 97.5]), diffs.mean()

ci, boot_mean = bootstrap_diff(x.to_numpy(), y.to_numpy())

print('summary')
print(summary)
print('welch_t', t_stat, p_val)
print('cohen_d', cohen_d)
print('mannwhitney_u', u_stat, u_p)
print('mean_diff', x.mean() - y.mean())
print('boot_ci', ci)

