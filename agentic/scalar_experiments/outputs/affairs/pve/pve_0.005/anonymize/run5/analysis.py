import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('affairs.csv')

# Map children yes/no to binary
children = df['feature6'].map({'yes': 1, 'no': 0})

df = df.assign(children=children)

# Summary statistics
summary = df.groupby('children')['feature2'].agg(['count', 'mean', 'std', 'median'])

# Two-sample t-test (Welch)
child_yes = df.loc[df['children'] == 1, 'feature2'].astype(float)
child_no = df.loc[df['children'] == 0, 'feature2'].astype(float)

t_stat, p_val = stats.ttest_ind(child_yes, child_no, equal_var=False, nan_policy='omit')

# Nonparametric test (Mann-Whitney U)
try:
    u_stat, u_p = stats.mannwhitneyu(child_yes, child_no, alternative='two-sided')
except ValueError:
    u_stat, u_p = np.nan, np.nan

# Effect size (Cohen's d)
mean_diff = child_yes.mean() - child_no.mean()
pooled_sd = np.sqrt(((child_yes.std(ddof=1) ** 2) + (child_no.std(ddof=1) ** 2)) / 2)
cohens_d = mean_diff / pooled_sd if pooled_sd != 0 else np.nan

# OLS regression: unadjusted and adjusted
# Use categorical for gender feature3

model_unadj = smf.ols('feature2 ~ children', data=df).fit(cov_type='HC3')

model_adj = smf.ols(
    'feature2 ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10',
    data=df
).fit(cov_type='HC3')

# Print results
print('Group summary by children')
print(summary)
print('\nMean difference (children=yes minus no):', mean_diff)
print('Cohen\'s d:', cohens_d)
print('\nWelch t-test: t =', t_stat, 'p =', p_val)
print('Mann-Whitney U: U =', u_stat, 'p =', u_p)

print('\nUnadjusted OLS coef for children (yes=1):')
print(model_unadj.params['children'], 'p =', model_unadj.pvalues['children'])
print('\nAdjusted OLS coef for children (yes=1):')
print(model_adj.params['children'], 'p =', model_adj.pvalues['children'])

# Save key stats to a json-like dict printed
out = {
    'summary': summary.to_dict(),
    'mean_diff': float(mean_diff),
    'cohens_d': float(cohens_d) if not np.isnan(cohens_d) else None,
    't_stat': float(t_stat),
    't_p': float(p_val),
    'u_stat': float(u_stat) if not np.isnan(u_stat) else None,
    'u_p': float(u_p) if not np.isnan(u_p) else None,
    'unadj_coef': float(model_unadj.params['children']),
    'unadj_p': float(model_unadj.pvalues['children']),
    'adj_coef': float(model_adj.params['children']),
    'adj_p': float(model_adj.pvalues['children']),
}

import json
print('\nKEY_STATS_JSON')
print(json.dumps(out, indent=2))
