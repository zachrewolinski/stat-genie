import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('affairs.csv')

# Ensure expected columns
required = ['feature2','feature6']
for c in required:
    if c not in df.columns:
        raise SystemExit(f'Missing {c}')

# feature2: frequency of affairs (numeric)
# feature6: children yes/no

# clean
sub = df[['feature2','feature6']].dropna()

# normalize yes/no
sub['feature6'] = sub['feature6'].astype(str).str.lower().str.strip()

# groups
grp = {k: v['feature2'].values for k, v in sub.groupby('feature6')}

if 'yes' not in grp or 'no' not in grp:
    raise SystemExit(f'Groups found: {list(grp.keys())}')

yes = grp['yes']
no = grp['no']

# Means
mean_yes = yes.mean()
mean_no = no.mean()

# Proportion any affairs
prop_yes = (yes > 0).mean()
prop_no = (no > 0).mean()

# Two-sample t-test (unequal var)
tstat, pval = stats.ttest_ind(no, yes, equal_var=False)

# Effect size (Cohen's d)
# pooled std with unequal sizes
n1, n2 = len(no), len(yes)
var1, var2 = no.var(ddof=1), yes.var(ddof=1)
pooled = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

d = (mean_no - mean_yes) / pooled if pooled > 0 else 0.0

# Proportion test (z test for two proportions)
count_no = (no > 0).sum()
count_yes = (yes > 0).sum()
prop_pooled = (count_no + count_yes) / (n1 + n2)

# standard error
se = np.sqrt(prop_pooled * (1-prop_pooled) * (1/n1 + 1/n2))
if se > 0:
    z = (prop_no - prop_yes) / se
    p_prop = 2*(1-stats.norm.cdf(abs(z)))
else:
    z = 0.0
    p_prop = 1.0

print('n_no', n1, 'n_yes', n2)
print('mean_no', mean_no, 'mean_yes', mean_yes, 'diff_no_minus_yes', mean_no-mean_yes)
print('prop_any_no', prop_no, 'prop_any_yes', prop_yes, 'diff', prop_no-prop_yes)
print('ttest_p', pval, 'tstat', tstat, 'cohen_d', d)
print('prop_z', z, 'prop_p', p_prop)
