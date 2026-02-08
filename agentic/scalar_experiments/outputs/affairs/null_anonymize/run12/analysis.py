import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('affairs.csv')

# Identify columns
children_col = 'feature6'
affairs_col = 'feature2'

# Clean
df = df[[children_col, affairs_col]].dropna()

# Ensure categories
df[children_col] = df[children_col].astype(str).str.lower()

# Split
with_children = df[df[children_col] == 'yes'][affairs_col].astype(float)
without_children = df[df[children_col] == 'no'][affairs_col].astype(float)

# Summary stats
mean_with = with_children.mean()
mean_without = without_children.mean()
median_with = with_children.median()
median_without = without_children.median()

# Proportion any affair (>0)
prop_with = (with_children > 0).mean()
prop_without = (without_children > 0).mean()

# Effect sizes
# Cohen's d
n1, n2 = len(with_children), len(without_children)
var1, var2 = with_children.var(ddof=1), without_children.var(ddof=1)
pooled_sd = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
cohen_d = (mean_with - mean_without) / pooled_sd if pooled_sd > 0 else np.nan

# Welch t-test
t_stat, p_val = stats.ttest_ind(with_children, without_children, equal_var=False)

# Print results
print('n_with', n1, 'n_without', n2)
print('mean_with', mean_with, 'mean_without', mean_without)
print('median_with', median_with, 'median_without', median_without)
print('prop_with', prop_with, 'prop_without', prop_without)
print('cohen_d', cohen_d)
print('t_stat', t_stat, 'p_val', p_val)

# Decide scalar: positive means children decrease affairs -> mean_with < mean_without
# We'll map effect size and proportion difference into score.
# Use difference in means and props for context.
mean_diff = mean_without - mean_with  # positive means decrease with children
prop_diff = prop_without - prop_with  # positive means decrease with children

# Normalize: small effect (d ~0.2) -> 20, medium (0.5)->50, large(0.8)->80
if np.isnan(cohen_d):
    base = 0
else:
    d = abs(cohen_d)
    if d < 0.1:
        base = 5
    elif d < 0.2:
        base = 10
    elif d < 0.35:
        base = 20
    elif d < 0.5:
        base = 35
    elif d < 0.65:
        base = 50
    elif d < 0.8:
        base = 65
    else:
        base = 80

# adjust by p-value (we only reduce if not significant)
if p_val > 0.2:
    base *= 0.5
elif p_val > 0.1:
    base *= 0.7
elif p_val > 0.05:
    base *= 0.85

# adjust by direction
score = base if mean_diff > 0 else -base

# additional small adjustment from prop_diff magnitude (cap +/-10)
score += max(-10, min(10, prop_diff * 50))

# clamp and round
score = int(round(max(-100, min(100, score))))
print('score', score)
