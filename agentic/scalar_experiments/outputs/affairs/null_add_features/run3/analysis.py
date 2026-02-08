import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('affairs.csv')

# Ensure expected columns
if 'children' not in df.columns or 'affairs' not in df.columns:
    raise SystemExit('Missing required columns')

# normalize children values
children = df['children'].astype(str).str.strip().str.lower()

# Use only rows with children yes/no
mask = children.isin(['yes', 'no']) & df['affairs'].notna()
sub = df.loc[mask].copy()
sub['children'] = children[mask]

# Binary affair indicator
sub['any_affair'] = (sub['affairs'] > 0).astype(int)

# group stats
stats_table = sub.groupby('children').agg(
    n=('affairs','size'),
    mean_affairs=('affairs','mean'),
    median_affairs=('affairs','median'),
    prop_any=('any_affair','mean')
)

# Difference in means (no - yes)
mean_no = stats_table.loc['no','mean_affairs']
mean_yes = stats_table.loc['yes','mean_affairs']
prop_no = stats_table.loc['no','prop_any']
prop_yes = stats_table.loc['yes','prop_any']

mean_diff = mean_no - mean_yes
prop_diff = prop_no - prop_yes

# t-test for affairs counts (Welch)
no_vals = sub.loc[sub['children']=='no','affairs']
yes_vals = sub.loc[sub['children']=='yes','affairs']

t_stat, p_val = stats.ttest_ind(no_vals, yes_vals, equal_var=False, nan_policy='omit')

# two-proportion z-test for any affair
# compute pooled
n_no = len(no_vals)
n_yes = len(yes_vals)
count_no = sub.loc[sub['children']=='no','any_affair'].sum()
count_yes = sub.loc[sub['children']=='yes','any_affair'].sum()

p1 = count_no / n_no
p2 = count_yes / n_yes
p_pool = (count_no + count_yes) / (n_no + n_yes)

se = np.sqrt(p_pool * (1 - p_pool) * (1/n_no + 1/n_yes))
if se > 0:
    z = (p1 - p2) / se
    p_val_prop = 2 * (1 - stats.norm.cdf(abs(z)))
else:
    z = np.nan
    p_val_prop = np.nan

# effect sizes
# Cohen's d for affairs counts
mean1 = mean_no
mean2 = mean_yes
sd1 = no_vals.std(ddof=1)
sd2 = yes_vals.std(ddof=1)

# pooled SD for d
s_pooled = np.sqrt(((n_no - 1)*sd1**2 + (n_yes - 1)*sd2**2) / (n_no + n_yes - 2))
if s_pooled > 0:
    d = (mean1 - mean2) / s_pooled
else:
    d = np.nan

# Risk ratio for any affair
risk_ratio = (p1 / p2) if p2 > 0 else np.nan

output = {
    'stats_table': stats_table,
    'mean_diff_no_minus_yes': mean_diff,
    'prop_diff_no_minus_yes': prop_diff,
    't_stat_affairs': t_stat,
    'p_val_affairs': p_val,
    'z_prop': z,
    'p_val_prop': p_val_prop,
    'cohens_d': d,
    'risk_ratio_any_affair': risk_ratio,
    'n_no': n_no,
    'n_yes': n_yes,
    'p1_no': p1,
    'p2_yes': p2,
}

# Save summary for manual review
stats_table.to_csv('children_affairs_stats.csv')

for k,v in output.items():
    if k == 'stats_table':
        continue
    print(f"{k}: {v}")

