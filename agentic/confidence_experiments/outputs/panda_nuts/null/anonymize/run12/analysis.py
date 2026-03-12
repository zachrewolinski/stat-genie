import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('panda_nuts.csv')

# Compute efficiency as nuts opened per second
# Avoid division by zero (duration min is 2.5 in metadata, but safeguard anyway)
df = df.copy()
df['efficiency'] = df['feature5'] / df['feature6']

# Basic cleaning / types
# feature2 = age, feature3 = sex, feature7 = help
df['feature3'] = df['feature3'].astype('category')
df['feature7'] = df['feature7'].astype('category')

# Regression: efficiency ~ age + sex + help
model = smf.ols('efficiency ~ feature2 + C(feature3) + C(feature7)', data=df).fit(cov_type='HC3')

# Spearman correlation for age vs efficiency
spearman_r, spearman_p = stats.spearmanr(df['feature2'], df['efficiency'])

# Group comparisons for sex and help (Mann-Whitney U)
# Sex groups
sex_groups = {k: v['efficiency'].values for k, v in df.groupby('feature3')}
sex_u = sex_p = None
if len(sex_groups) == 2:
    a, b = list(sex_groups.values())
    sex_u, sex_p = stats.mannwhitneyu(a, b, alternative='two-sided')

# Help groups
help_groups = {k: v['efficiency'].values for k, v in df.groupby('feature7')}
help_u = help_p = None
if len(help_groups) == 2:
    a, b = list(help_groups.values())
    help_u, help_p = stats.mannwhitneyu(a, b, alternative='two-sided')

# Summary stats
summary = {
    'n': int(df.shape[0]),
    'efficiency_mean': float(df['efficiency'].mean()),
    'efficiency_median': float(df['efficiency'].median()),
    'efficiency_std': float(df['efficiency'].std()),
    'age_mean': float(df['feature2'].mean()),
    'age_std': float(df['feature2'].std()),
}

# Model coefficients and p-values
params = model.params.to_dict()
pvalues = model.pvalues.to_dict()

# R-squared
r2 = float(model.rsquared)

results = {
    'summary': summary,
    'model_params': params,
    'model_pvalues': pvalues,
    'model_r2': r2,
    'spearman_age_efficiency': {'r': float(spearman_r), 'p': float(spearman_p)},
    'mannwhitney_sex': {'u': None if sex_u is None else float(sex_u), 'p': None if sex_p is None else float(sex_p)},
    'mannwhitney_help': {'u': None if help_u is None else float(help_u), 'p': None if help_p is None else float(help_p)},
}

print(json.dumps(results, indent=2, sort_keys=True))
