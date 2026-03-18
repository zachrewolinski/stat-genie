import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


df = pd.read_csv('affairs.csv')

# Map columns to meaningful names
colmap = {
    'feature2': 'affairs',
    'feature3': 'gender',
    'feature4': 'age',
    'feature5': 'years_married',
    'feature6': 'children',
    'feature7': 'religiousness',
    'feature8': 'education',
    'feature9': 'occupation',
    'feature10': 'marriage_rating',
}

df = df.rename(columns=colmap)

# Basic group stats
children_groups = df['children'].unique()

summary = df.groupby('children')['affairs'].agg(['count', 'mean', 'median', 'std'])
summary['prop_any_affair'] = df.groupby('children')['affairs'].apply(lambda s: (s > 0).mean())

# Group arrays
aff_yes = df[df['children'] == 'yes']['affairs']
aff_no = df[df['children'] == 'no']['affairs']

# Welch t-test
welch_t = stats.ttest_ind(aff_yes, aff_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U test
mwu = stats.mannwhitneyu(aff_yes, aff_no, alternative='two-sided')

# Effect sizes
mean_diff = aff_yes.mean() - aff_no.mean()  # children yes minus no
median_diff = aff_yes.median() - aff_no.median()

# Logistic regression: any affair
# Create indicator
_df = df.copy()
_df['affair_any'] = (_df['affairs'] > 0).astype(int)

logit_simple = smf.logit('affair_any ~ C(children)', data=_df).fit(disp=0)
logit_ctrl = smf.logit(
    'affair_any ~ C(children) + C(gender) + age + years_married + religiousness + education + occupation + marriage_rating',
    data=_df
).fit(disp=0)

# OLS on affair frequency
ols_simple = smf.ols('affairs ~ C(children)', data=_df).fit(cov_type='HC3')
ols_ctrl = smf.ols(
    'affairs ~ C(children) + C(gender) + age + years_married + religiousness + education + occupation + marriage_rating',
    data=_df
).fit(cov_type='HC3')

# Collect key results
results = {
    'summary': summary,
    'mean_diff_children_yes_minus_no': mean_diff,
    'median_diff_children_yes_minus_no': median_diff,
    'welch_t_stat': welch_t.statistic,
    'welch_t_pvalue': welch_t.pvalue,
    'mwu_stat': mwu.statistic,
    'mwu_pvalue': mwu.pvalue,
    'logit_simple_params': logit_simple.params,
    'logit_simple_pvalues': logit_simple.pvalues,
    'logit_ctrl_params': logit_ctrl.params,
    'logit_ctrl_pvalues': logit_ctrl.pvalues,
    'ols_simple_params': ols_simple.params,
    'ols_simple_pvalues': ols_simple.pvalues,
    'ols_ctrl_params': ols_ctrl.params,
    'ols_ctrl_pvalues': ols_ctrl.pvalues,
}

# Print results in a readable way
print('Group summary:\n', summary)
print('\nMean difference (yes - no):', mean_diff)
print('Median difference (yes - no):', median_diff)
print('\nWelch t-test:', welch_t)
print('Mann-Whitney U:', mwu)

print('\nLogit simple params:\n', logit_simple.params)
print('Logit simple pvalues:\n', logit_simple.pvalues)

print('\nLogit controlled params:\n', logit_ctrl.params)
print('Logit controlled pvalues:\n', logit_ctrl.pvalues)

print('\nOLS simple params:\n', ols_simple.params)
print('OLS simple pvalues:\n', ols_simple.pvalues)

print('\nOLS controlled params:\n', ols_ctrl.params)
print('OLS controlled pvalues:\n', ols_ctrl.pvalues)

