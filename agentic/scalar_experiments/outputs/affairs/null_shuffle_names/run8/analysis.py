import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('affairs.csv')

# Column mapping based on info.json descriptions
# 'religiousness' column actually indicates whether there are children in the marriage (yes/no)
# 'age' column actually contains the affairs frequency scale
children_col = 'religiousness'
affairs_col = 'age'

# Clean/encode
children = df[children_col].astype(str).str.strip().str.lower()
# Keep only yes/no rows
mask = children.isin(['yes', 'no'])
if not mask.all():
    df = df.loc[mask].copy()
    children = children[mask]

affairs = pd.to_numeric(df[affairs_col], errors='coerce')
valid = affairs.notna()
children = children[valid]
affairs = affairs[valid]

# Group stats
summary = {}
for label in ['yes', 'no']:
    grp = affairs[children == label]
    summary[label] = {
        'n': int(grp.shape[0]),
        'mean': float(grp.mean()) if grp.shape[0] else float('nan'),
        'median': float(grp.median()) if grp.shape[0] else float('nan'),
        'prop_any': float((grp > 0).mean()) if grp.shape[0] else float('nan'),
    }

# Differences
mean_yes = summary['yes']['mean']
mean_no = summary['no']['mean']
prop_yes = summary['yes']['prop_any']
prop_no = summary['no']['prop_any']

diff_mean = mean_no - mean_yes  # positive means children associated with fewer affairs

diff_prop = prop_no - prop_yes

# t-test (Welch)
try:
    tstat, pval = stats.ttest_ind(
        affairs[children == 'no'],
        affairs[children == 'yes'],
        equal_var=False,
        nan_policy='omit',
    )
except Exception:
    tstat, pval = float('nan'), float('nan')

# Mann-Whitney U test
try:
    ustat, pval_u = stats.mannwhitneyu(
        affairs[children == 'no'],
        affairs[children == 'yes'],
        alternative='two-sided',
    )
except Exception:
    ustat, pval_u = float('nan'), float('nan')

# Effect size (Cohen's d)
def cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float('nan')
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = np.sqrt(((a.size - 1) * va + (b.size - 1) * vb) / (a.size + b.size - 2))
    if pooled == 0:
        return float('nan')
    return (a.mean() - b.mean()) / pooled

cohen_d = cohens_d(affairs[children == 'no'], affairs[children == 'yes'])

print('Summary by children (yes/no):')
for k, v in summary.items():
    print(k, v)
print('diff_mean (no - yes):', diff_mean)
print('diff_prop (no - yes):', diff_prop)
print('t-test t, p:', tstat, pval)
print('mannwhitney u, p:', ustat, pval_u)
print('cohen_d (no - yes):', cohen_d)
