import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('affairs.csv')

# Map variables based on metadata/value patterns
# children indicator is stored in column 'religiousness' (yes/no)
children = df['religiousness'].map({'no': 0, 'yes': 1})

# Affairs frequency appears to be stored in 'education' column but scaled by 1000
# (range 0.004-9.029). Scaling doesn't affect inference, but we use scaled for interpretability.
affairs = df['education'] / 1000.0

# Group stats
stats_by_child = affairs.groupby(children).agg(['count', 'mean', 'std', 'median'])

# Welch t-test
no_group = affairs[children == 0]
yes_group = affairs[children == 1]

t_stat, t_p = stats.ttest_ind(yes_group, no_group, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (two-sided)
try:
    u_stat, u_p = stats.mannwhitneyu(yes_group, no_group, alternative='two-sided')
except ValueError:
    u_stat, u_p = np.nan, np.nan

# Effect size (Cohen's d) using pooled std (Hedges g not necessary for descriptive)
mean_diff = yes_group.mean() - no_group.mean()
pooled_std = np.sqrt(((yes_group.std(ddof=1) ** 2) + (no_group.std(ddof=1) ** 2)) / 2)
cohen_d = mean_diff / pooled_std if pooled_std > 0 else np.nan

print('Group stats (children: 0=no, 1=yes)')
print(stats_by_child)
print('\nWelch t-test: t=%.4f p=%.6f' % (t_stat, t_p))
print('Mann-Whitney U: U=%.4f p=%.6f' % (u_stat, u_p))
print('Mean difference (yes-no): %.6f' % mean_diff)
print('Cohen d: %.4f' % cohen_d)

# Also compute simple Spearman correlation
rho, rho_p = stats.spearmanr(children, affairs)
print('Spearman rho: %.4f p=%.6f' % (rho, rho_p))
