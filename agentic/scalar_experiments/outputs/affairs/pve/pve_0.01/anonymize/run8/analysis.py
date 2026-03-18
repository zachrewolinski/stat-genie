import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


df = pd.read_csv('affairs.csv')

# Binary indicator: 1 = children yes, 0 = no
children = df['feature6'].str.lower().map({'yes': 1, 'no': 0})
df = df.assign(children=children)

# Group stats
grp = df.groupby('children')['feature2']
means = grp.mean()
stds = grp.std(ddof=1)
counts = grp.count()

# Welch t-test
vals_yes = df.loc[df['children'] == 1, 'feature2']
vals_no = df.loc[df['children'] == 0, 'feature2']
t_stat, t_p = stats.ttest_ind(vals_yes, vals_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (two-sided)
try:
    u_stat, u_p = stats.mannwhitneyu(vals_yes, vals_no, alternative='two-sided')
except ValueError:
    u_stat, u_p = np.nan, np.nan

# Cohen's d (pooled SD)
mean_diff = means.loc[1] - means.loc[0]
pooled_sd = np.sqrt(((counts.loc[1]-1)*stds.loc[1]**2 + (counts.loc[0]-1)*stds.loc[0]**2) / (counts.loc[1]+counts.loc[0]-2))
cohen_d = mean_diff / pooled_sd if pooled_sd != 0 else np.nan

# Regression with controls
model = smf.ols(
    'feature2 ~ children + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10',
    data=df
).fit(cov_type='HC3')

coef = model.params['children']
pval = model.pvalues['children']
ci_low, ci_high = model.conf_int().loc['children']

results = {
    'counts': counts.to_dict(),
    'means': means.to_dict(),
    'stds': stds.to_dict(),
    'mean_diff_yes_minus_no': float(mean_diff),
    't_test_p': float(t_p),
    'u_test_p': float(u_p),
    'cohen_d': float(cohen_d),
    'regression_children_coef': float(coef),
    'regression_children_p': float(pval),
    'regression_children_ci': [float(ci_low), float(ci_high)],
    'r2': float(model.rsquared),
    'n': int(model.nobs),
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
