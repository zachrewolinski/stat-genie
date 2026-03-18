import json
import numpy as np
import pandas as pd
from scipy import stats

# Load data
_df = pd.read_csv('affairs.csv')

# According to info.json metadata, column 'religiousness' indicates whether there are children
# and column 'age' captures frequency of extramarital affairs.
children_col = 'religiousness'
affairs_col = 'age'

# Map children to binary
children_map = {'yes': 1, 'no': 0}
children = _df[children_col].map(children_map)

# Outcome
affairs = _df[affairs_col].astype(float)

# Drop missing
mask = children.notna() & affairs.notna()
children = children[mask]
affairs = affairs[mask]

# Group stats
grp_yes = affairs[children == 1]
grp_no = affairs[children == 0]

summary = {
    'n_yes': int(grp_yes.shape[0]),
    'n_no': int(grp_no.shape[0]),
    'mean_yes': float(grp_yes.mean()),
    'mean_no': float(grp_no.mean()),
    'median_yes': float(grp_yes.median()),
    'median_no': float(grp_no.median()),
    'std_yes': float(grp_yes.std(ddof=1)),
    'std_no': float(grp_no.std(ddof=1)),
}

# Welch's t-test
welch = stats.ttest_ind(grp_yes, grp_no, equal_var=False)

# Mann-Whitney U (two-sided)
try:
    mw = stats.mannwhitneyu(grp_yes, grp_no, alternative='two-sided')
except TypeError:
    # Older SciPy versions return only statistic and p-value
    mw = stats.mannwhitneyu(grp_yes, grp_no)

# Effect size (Cohen's d)
mean_diff = summary['mean_yes'] - summary['mean_no']
var_yes = grp_yes.var(ddof=1)
var_no = grp_no.var(ddof=1)
# Pooled SD for Cohen's d
pooled_sd = np.sqrt(((grp_yes.shape[0] - 1) * var_yes + (grp_no.shape[0] - 1) * var_no) / (grp_yes.shape[0] + grp_no.shape[0] - 2))
cohen_d = mean_diff / pooled_sd if pooled_sd != 0 else np.nan

# 95% CI for mean difference (Welch)
# Standard error for difference
se_diff = np.sqrt(var_yes / grp_yes.shape[0] + var_no / grp_no.shape[0])
# Welch-Satterthwaite df
df_num = (var_yes / grp_yes.shape[0] + var_no / grp_no.shape[0]) ** 2
df_den = ((var_yes / grp_yes.shape[0]) ** 2) / (grp_yes.shape[0] - 1) + ((var_no / grp_no.shape[0]) ** 2) / (grp_no.shape[0] - 1)
df_welch = df_num / df_den
crit = stats.t.ppf(0.975, df_welch)
ci_low = mean_diff - crit * se_diff
ci_high = mean_diff + crit * se_diff

results = {
    'summary': summary,
    'mean_diff_yes_minus_no': float(mean_diff),
    'welch_t': float(welch.statistic),
    'welch_p': float(welch.pvalue),
    'mannwhitney_u': float(getattr(mw, 'statistic', mw[0])),
    'mannwhitney_p': float(getattr(mw, 'pvalue', mw[1])),
    'cohen_d': float(cohen_d),
    'ci95_diff': [float(ci_low), float(ci_high)],
}

print(json.dumps(results, indent=2))
