import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv("affairs.csv")

# Map children yes/no
if df['feature6'].dtype == object:
    children = df['feature6'].str.lower().str.strip()
else:
    children = df['feature6']

# Affairs frequency
affairs = df['feature2'].astype(float)

# groups
mask_yes = children == 'yes'
mask_no = children == 'no'

# Basic stats
summary = {}
for label, mask in [('yes', mask_yes), ('no', mask_no)]:
    vals = affairs[mask]
    summary[label] = {
        'n': int(vals.shape[0]),
        'mean': float(vals.mean()),
        'median': float(vals.median()),
        'p_gt0': float((vals > 0).mean()),
        'p_ge7': float((vals >= 7).mean()),
        'std': float(vals.std(ddof=1)) if vals.shape[0] > 1 else float('nan'),
    }

# Difference in means
mean_yes = summary['yes']['mean']
mean_no = summary['no']['mean']
mean_diff = mean_yes - mean_no

# Difference in proportions with any affairs
p_yes = summary['yes']['p_gt0']
p_no = summary['no']['p_gt0']
p_diff = p_yes - p_no

# Welch t-test for mean difference
vals_yes = affairs[mask_yes]
vals_no = affairs[mask_no]

t_res = stats.ttest_ind(vals_yes, vals_no, equal_var=False, nan_policy='omit')

# Two-proportion z-test for any affairs
count_yes = int((vals_yes > 0).sum())
count_no = int((vals_no > 0).sum())

n_yes = int(mask_yes.sum())
n_no = int(mask_no.sum())

p_pool = (count_yes + count_no) / (n_yes + n_no)
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_yes + 1 / n_no))
if se_pool == 0:
    z = np.nan
    pval_prop = np.nan
else:
    z = (p_yes - p_no) / se_pool
    pval_prop = 2 * (1 - stats.norm.cdf(abs(z)))

# Effect size (Cohen's d)
# Using pooled SD
sd_pooled = np.sqrt(((n_yes - 1) * vals_yes.var(ddof=1) + (n_no - 1) * vals_no.var(ddof=1)) / (n_yes + n_no - 2))
cohen_d = (mean_yes - mean_no) / sd_pooled if sd_pooled > 0 else np.nan

out = {
    'summary': summary,
    'mean_diff': mean_diff,
    'prop_diff': p_diff,
    't_stat': float(t_res.statistic),
    't_pval': float(t_res.pvalue),
    'prop_z': float(z),
    'prop_pval': float(pval_prop),
    'cohen_d': float(cohen_d),
}

print(out)
