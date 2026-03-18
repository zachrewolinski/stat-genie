import pandas as pd
import numpy as np
from scipy import stats

# Load data
_df = pd.read_csv('affairs.csv')

# Map columns
# feature2: affairs frequency numeric
# feature6: children yes/no

# Basic cleaning
# Ensure feature6 is lower-case strings
children = _df['feature6'].astype(str).str.lower()
# Ensure feature2 numeric
affairs = pd.to_numeric(_df['feature2'], errors='coerce')

# Drop missing
mask = children.isin(['yes','no']) & affairs.notna()
children = children[mask]
affairs = affairs[mask]

# Groups
grp_yes = affairs[children == 'yes']
grp_no = affairs[children == 'no']

# Summary stats
summary = {
    'n_yes': grp_yes.shape[0],
    'n_no': grp_no.shape[0],
    'mean_yes': grp_yes.mean(),
    'mean_no': grp_no.mean(),
    'median_yes': grp_yes.median(),
    'median_no': grp_no.median(),
    'zero_prop_yes': (grp_yes==0).mean(),
    'zero_prop_no': (grp_no==0).mean(),
}

# t-test (Welch)
t_stat, t_p = stats.ttest_ind(grp_yes, grp_no, equal_var=False)

# Mann-Whitney U (non-param)
u_stat, u_p = stats.mannwhitneyu(grp_yes, grp_no, alternative='two-sided')

# Effect size Cohen's d
mean_diff = summary['mean_yes'] - summary['mean_no']
# pooled std
s1 = grp_yes.std(ddof=1)
s2 = grp_no.std(ddof=1)
sp = np.sqrt(((grp_yes.shape[0]-1)*s1**2 + (grp_no.shape[0]-1)*s2**2) / (grp_yes.shape[0]+grp_no.shape[0]-2))
cohen_d = mean_diff / sp if sp != 0 else np.nan

# Rank-biserial effect size for Mann-Whitney
n1 = grp_yes.shape[0]
n2 = grp_no.shape[0]
rank_biserial = 1 - (2*u_stat)/(n1*n2)

print('summary', summary)
print('t_stat', t_stat, 't_p', t_p)
print('u_stat', u_stat, 'u_p', u_p)
print('cohen_d', cohen_d)
print('rank_biserial', rank_biserial)
