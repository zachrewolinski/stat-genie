import pandas as pd
import numpy as np
from scipy import stats

# Load data
path = 'affairs.csv'
df = pd.read_csv(path)

# Identify columns
# feature2: affairs frequency; feature6: children yes/no

affairs = df['feature2']
children = df['feature6']

# Basic stats by children
summary = df.groupby('feature6')['feature2'].agg(['count','mean','median','std'])

# Proportion with any affairs
any_affair = (affairs > 0).astype(int)
ct = pd.crosstab(children, any_affair)

# Chi-square test of independence
chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)

# Effect size for 2x2: phi coefficient
n = ct.to_numpy().sum()
phi = np.sqrt(chi2 / n)

# Difference in proportion (children yes vs no) for any affair
prop_yes = ct.loc['yes', 1] / ct.loc['yes'].sum() if 'yes' in ct.index else np.nan
prop_no = ct.loc['no', 1] / ct.loc['no'].sum() if 'no' in ct.index else np.nan
prop_diff = prop_yes - prop_no

# Mann-Whitney U test for distribution of affair frequency
# Use two-sided because question is directional but we can interpret direction
x_yes = df.loc[df['feature6']=='yes', 'feature2']
x_no = df.loc[df['feature6']=='no', 'feature2']

u_stat, p_mwu = stats.mannwhitneyu(x_yes, x_no, alternative='two-sided')

# Compute Cliff's delta effect size
# Efficient computation using ranks
# Based on: delta = (2*U)/(n1*n2) - 1, where U is for x_yes > x_no
n1, n2 = len(x_yes), len(x_no)
cliffs_delta = (2 * u_stat) / (n1 * n2) - 1

# Also compare means with Welch t-test (robust to unequal variances)
t_stat, p_t = stats.ttest_ind(x_yes, x_no, equal_var=False)

# Save results
results = {
    'summary_by_children': summary.to_dict(),
    'crosstab_any_affair': ct.to_dict(),
    'chi2': chi2,
    'p_chi2': p_chi2,
    'phi': phi,
    'prop_yes': prop_yes,
    'prop_no': prop_no,
    'prop_diff_yes_minus_no': prop_diff,
    'mannwhitney_u': u_stat,
    'p_mwu': p_mwu,
    'cliffs_delta': cliffs_delta,
    't_stat': t_stat,
    'p_t': p_t,
    'n_yes': n1,
    'n_no': n2,
}

print(results)
