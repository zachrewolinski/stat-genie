import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('affairs.csv')

# normalize children column to lower-case strings
children = df['children'].astype(str).str.strip().str.lower()

# outcomes: affairs count and any affairs indicator
affairs = pd.to_numeric(df['affairs'], errors='coerce')
any_affairs = (affairs > 0).astype(int)

# groups
mask_yes = children.isin(['yes', 'y', '1', 'true', 't'])
mask_no = children.isin(['no', 'n', '0', 'false', 'f'])

# restrict to valid yes/no
valid = mask_yes | mask_no

df_valid = df[valid].copy()
children_valid = children[valid]
affairs_valid = affairs[valid]
any_affairs_valid = any_affairs[valid]

mask_yes = children_valid.isin(['yes', 'y', '1', 'true', 't'])
mask_no = children_valid.isin(['no', 'n', '0', 'false', 'f'])

g1 = affairs_valid[mask_yes]
g0 = affairs_valid[mask_no]

p1 = any_affairs_valid[mask_yes]
p0 = any_affairs_valid[mask_no]

# summary stats
summary = {
    'n_yes': int(mask_yes.sum()),
    'n_no': int(mask_no.sum()),
    'mean_affairs_yes': float(g1.mean()),
    'mean_affairs_no': float(g0.mean()),
    'median_affairs_yes': float(g1.median()),
    'median_affairs_no': float(g0.median()),
    'any_affairs_rate_yes': float(p1.mean()),
    'any_affairs_rate_no': float(p0.mean()),
}

# effect sizes and tests
# Welch's t-test on affairs counts
try:
    t_stat, t_p = stats.ttest_ind(g1, g0, equal_var=False, nan_policy='omit')
except Exception:
    t_stat, t_p = np.nan, np.nan

# Mann-Whitney U for non-normal
try:
    u_stat, u_p = stats.mannwhitneyu(g1, g0, alternative='two-sided')
except Exception:
    u_stat, u_p = np.nan, np.nan

# difference in proportions (any affairs)
# compute z-test for proportions
n1 = mask_yes.sum()
n0 = mask_no.sum()

p1_rate = p1.mean()
p0_rate = p0.mean()

# pooled proportion
p_pool = (p1.sum() + p0.sum()) / (n1 + n0)

se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n0))
if se == 0:
    z_stat = np.nan
    z_p = np.nan
else:
    z_stat = (p1_rate - p0_rate) / se
    z_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))

summary.update({
    't_stat': float(t_stat),
    't_p': float(t_p),
    'u_stat': float(u_stat),
    'u_p': float(u_p),
    'prop_z': float(z_stat),
    'prop_p': float(z_p),
})

# Cohen's d for affairs counts
# use pooled std
s1 = g1.std(ddof=1)
s0 = g0.std(ddof=1)
sp = np.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2) / (n1 + n0 - 2))
cohen_d = (g1.mean() - g0.mean()) / sp if sp != 0 else np.nan
summary['cohen_d'] = float(cohen_d)

# print
for k, v in summary.items():
    print(f"{k}: {v}")

# also show counts of children values
print('\nchildren value counts:')
print(children.value_counts(dropna=False))
