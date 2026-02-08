import pandas as pd
import numpy as np
from scipy import stats

# Load data
_df = pd.read_csv('affairs.csv')

# Identify columns by metadata assumptions
# feature2: affair frequency numeric
# feature6: children yes/no

affair = _df['feature2']
children = _df['feature6']

# Basic groups
yes = _df[children == 'yes']
no = _df[children == 'no']

results = {}

results['n_yes'] = len(yes)
results['n_no'] = len(no)

# Means
results['mean_affair_yes'] = yes['feature2'].mean()
results['mean_affair_no'] = no['feature2'].mean()

# Medians
results['median_affair_yes'] = yes['feature2'].median()
results['median_affair_no'] = no['feature2'].median()

# Proportion any affairs (>0)
results['prop_any_yes'] = (yes['feature2'] > 0).mean()
results['prop_any_no'] = (no['feature2'] > 0).mean()

# Effect sizes
# Cohen's d for mean difference
mean_diff = results['mean_affair_yes'] - results['mean_affair_no']
pooled_sd = np.sqrt(((len(yes)-1)*yes['feature2'].var(ddof=1) + (len(no)-1)*no['feature2'].var(ddof=1)) / (len(yes)+len(no)-2))
results['cohens_d'] = mean_diff / pooled_sd if pooled_sd > 0 else np.nan

# t-test (Welch)
try:
    tstat, pval = stats.ttest_ind(yes['feature2'], no['feature2'], equal_var=False, nan_policy='omit')
except Exception:
    tstat, pval = np.nan, np.nan
results['ttest_t'] = tstat
results['ttest_p'] = pval

# Mann-Whitney U test for distributional difference
try:
    ustat, pval_u = stats.mannwhitneyu(yes['feature2'], no['feature2'], alternative='two-sided')
except Exception:
    ustat, pval_u = np.nan, np.nan
results['mw_u'] = ustat
results['mw_p'] = pval_u

# Print results
for k, v in results.items():
    print(f"{k}: {v}")
