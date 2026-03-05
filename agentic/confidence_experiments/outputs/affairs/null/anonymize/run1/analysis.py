import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
file_path = 'affairs.csv'
df = pd.read_csv(file_path)

# Variables
children = df['feature6'].astype(str)
affairs = pd.to_numeric(df['feature2'], errors='coerce')

# Binary affair indicator
any_affair = (affairs > 0).astype(int)

# Group masks
mask_yes = children.str.lower() == 'yes'
mask_no = children.str.lower() == 'no'

# Descriptive stats
summary = {}
for label, mask in [('yes', mask_yes), ('no', mask_no)]:
    grp_affairs = affairs[mask].dropna()
    grp_any = any_affair[mask]
    summary[label] = {
        'n': int(mask.sum()),
        'mean_affairs': float(grp_affairs.mean()),
        'median_affairs': float(grp_affairs.median()),
        'prop_any_affair': float(grp_any.mean()),
        'std_affairs': float(grp_affairs.std(ddof=1))
    }

# Welch t-test for mean differences
# (affairs may be skewed but t-test ok with large n; we'll also do Mann-Whitney)

aff_yes = affairs[mask_yes].dropna()
aff_no = affairs[mask_no].dropna()

# Welch t-test
welch_t = stats.ttest_ind(aff_yes, aff_no, equal_var=False)

# Mann-Whitney U test (two-sided)
mann_u = stats.mannwhitneyu(aff_yes, aff_no, alternative='two-sided')

# Two-proportion z-test for any affair
# Use statsmodels
count_yes = any_affair[mask_yes].sum()
count_no = any_affair[mask_no].sum()

n_yes = mask_yes.sum()
n_no = mask_no.sum()

stat_prop, p_prop = sm.stats.proportions_ztest([count_yes, count_no], [n_yes, n_no])

# Effect sizes
# Cohen's d (using pooled SD)
mean_yes = aff_yes.mean()
mean_no = aff_no.mean()

sd_yes = aff_yes.std(ddof=1)
sd_no = aff_no.std(ddof=1)

pooled_sd = np.sqrt(((len(aff_yes)-1)*sd_yes**2 + (len(aff_no)-1)*sd_no**2) / (len(aff_yes)+len(aff_no)-2))
cohen_d = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan

# Risk difference and relative risk for any affair
p_yes = count_yes / n_yes
p_no = count_no / n_no
risk_diff = p_yes - p_no
rel_risk = p_yes / p_no if p_no > 0 else np.nan

results = {
    'summary': summary,
    'welch_t': {'stat': float(welch_t.statistic), 'p': float(welch_t.pvalue)},
    'mannwhitney': {'stat': float(mann_u.statistic), 'p': float(mann_u.pvalue)},
    'prop_test': {'stat': float(stat_prop), 'p': float(p_prop)},
    'effects': {
        'cohen_d': float(cohen_d),
        'risk_diff': float(risk_diff),
        'relative_risk': float(rel_risk)
    }
}

print(json.dumps(results, indent=2))
