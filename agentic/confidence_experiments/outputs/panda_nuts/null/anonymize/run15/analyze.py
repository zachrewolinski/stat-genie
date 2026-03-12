import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats


df = pd.read_csv('panda_nuts.csv')
# Rename for clarity
cols = {
    'feature1': 'id',
    'feature2': 'age',
    'feature3': 'sex',
    'feature4': 'hammer',
    'feature5': 'nuts_opened',
    'feature6': 'duration_sec',
    'feature7': 'help'
}
df = df.rename(columns=cols)

# Efficiency: nuts opened per second
# Avoid division issues (duration_sec > 0 by metadata)
df['efficiency'] = df['nuts_opened'] / df['duration_sec']

# Basic stats
summary = df[['age','nuts_opened','duration_sec','efficiency']].describe()

# Encode categorical
# Fit OLS model with categorical predictors
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit()

# Also check if age non-linear: add quadratic term
model_quad = smf.ols('efficiency ~ age + I(age**2) + C(sex) + C(help)', data=df).fit()

# Group comparisons
sex_groups = df.groupby('sex')['efficiency']
help_groups = df.groupby('help')['efficiency']

# t-tests (Welch)
sex_levels = list(sex_groups.groups.keys())
help_levels = list(help_groups.groups.keys())

sex_t = None
if len(sex_levels) == 2:
    g1 = sex_groups.get_group(sex_levels[0])
    g2 = sex_groups.get_group(sex_levels[1])
    sex_t = stats.ttest_ind(g1, g2, equal_var=False)

help_t = None
if len(help_levels) == 2:
    g1 = help_groups.get_group(help_levels[0])
    g2 = help_groups.get_group(help_levels[1])
    help_t = stats.ttest_ind(g1, g2, equal_var=False)

# Correlation age vs efficiency
age_corr = stats.pearsonr(df['age'], df['efficiency'])

# Save key results
results = {
    'n': len(df),
    'summary': summary.to_dict(),
    'model_params': model.params.to_dict(),
    'model_pvalues': model.pvalues.to_dict(),
    'model_r2': model.rsquared,
    'model_adj_r2': model.rsquared_adj,
    'model_f_pvalue': model.f_pvalue,
    'model_quad_pvalues': model_quad.pvalues.to_dict(),
    'model_quad_adj_r2': model_quad.rsquared_adj,
    'sex_levels': sex_levels,
    'help_levels': help_levels,
    'sex_t': None if sex_t is None else {'stat': float(sex_t.statistic), 'pvalue': float(sex_t.pvalue)},
    'help_t': None if help_t is None else {'stat': float(help_t.statistic), 'pvalue': float(help_t.pvalue)},
    'age_corr': {'r': float(age_corr.statistic if hasattr(age_corr,'statistic') else age_corr[0]), 'pvalue': float(age_corr.pvalue if hasattr(age_corr,'pvalue') else age_corr[1])},
}

import json
with open('analysis_results.json','w') as f:
    json.dump(results, f, indent=2)

print('Saved analysis_results.json')
