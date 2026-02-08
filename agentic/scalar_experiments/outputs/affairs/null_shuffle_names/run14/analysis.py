import pandas as pd
import numpy as np
from scipy import stats

# Load data
path = 'affairs.csv'
df = pd.read_csv(path)

# Identify key columns based on metadata shuffle reasoning
# 'age' column appears to encode extramarital affairs frequency (0-12 scale)
# 'religiousness' column is yes/no and appears to encode presence of children

# Clean
# ensure proper types

# Map children
if df['religiousness'].dtype == object:
    children = df['religiousness'].str.strip().str.lower()
else:
    children = df['religiousness']

# affairs frequency
affairs = pd.to_numeric(df['age'], errors='coerce')

mask = children.isin(['yes','no']) & affairs.notna()
sub = df.loc[mask].copy()
sub['children_yes'] = children[mask].str.lower().eq('yes')
sub['affairs_freq'] = affairs[mask]

# group stats
stats_group = sub.groupby('children_yes')['affairs_freq'].agg(['count','mean','median','std']).rename(index={False:'no_children', True:'children'})

# Welch t-test
no = sub.loc[~sub['children_yes'], 'affairs_freq']
yes = sub.loc[sub['children_yes'], 'affairs_freq']

t_stat, p_val = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')

# Cohen's d (Hedges g)
mean_diff = yes.mean() - no.mean()
# pooled SD for effect size
n1, n2 = yes.count(), no.count()
var1, var2 = yes.var(ddof=1), no.var(ddof=1)
pooled = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)) if (n1+n2-2) > 0 else np.nan
cohen_d = mean_diff / pooled if pooled and pooled > 0 else np.nan

# Mann-Whitney U
try:
    u_stat, u_p = stats.mannwhitneyu(yes, no, alternative='two-sided')
except Exception:
    u_stat, u_p = np.nan, np.nan

print('Group stats:\n', stats_group)
print('\nMean difference (children - no):', mean_diff)
print('Welch t-test:', t_stat, p_val)
print('Cohen d:', cohen_d)
print('Mann-Whitney U p:', u_p)
