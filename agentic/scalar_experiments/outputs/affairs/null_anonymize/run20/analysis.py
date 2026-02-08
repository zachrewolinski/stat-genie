import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('affairs.csv')

# basic cleanup
# children indicator
children_col = 'feature6'
engagement_col = 'feature2'

# ensure valid rows
sub = df[[children_col, engagement_col]].dropna()

# map children yes/no
sub[children_col] = sub[children_col].astype(str).str.strip().str.lower()

yes = sub[sub[children_col] == 'yes'][engagement_col].astype(float)
no = sub[sub[children_col] == 'no'][engagement_col].astype(float)

summary = {
    'n_yes': len(yes),
    'n_no': len(no),
    'mean_yes': yes.mean(),
    'mean_no': no.mean(),
    'median_yes': yes.median(),
    'median_no': no.median(),
    'prop_any_yes': (yes > 0).mean(),
    'prop_any_no': (no > 0).mean(),
}

# t-test on engagement frequency
if len(yes) > 1 and len(no) > 1:
    t_res = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')
else:
    t_res = None

# effect size: Cohen's d (unequal n, pooled std)
if len(yes) > 1 and len(no) > 1:
    var_yes = yes.var(ddof=1)
    var_no = no.var(ddof=1)
    n1, n2 = len(yes), len(no)
    pooled = ((n1 - 1) * var_yes + (n2 - 1) * var_no) / (n1 + n2 - 2)
    d = (yes.mean() - no.mean()) / np.sqrt(pooled) if pooled > 0 else np.nan
else:
    d = np.nan

# proportion test for any affair
count_yes = int((yes > 0).sum())
count_no = int((no > 0).sum())

# two-proportion z-test
if len(yes) > 0 and len(no) > 0:
    p1 = count_yes / len(yes)
    p2 = count_no / len(no)
    p_pool = (count_yes + count_no) / (len(yes) + len(no))
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / len(yes) + 1 / len(no)))
    z = (p1 - p2) / se if se > 0 else np.nan
    pval_prop = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
else:
    z = np.nan
    pval_prop = np.nan

print('Summary:', summary)
print('t-test:', t_res)
print('cohen_d_yes_minus_no:', d)
print('prop_test_z:', z, 'pval:', pval_prop)
