import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
path = 'affairs.csv'
df = pd.read_csv(path)

# Basic group stats
summary = df.groupby('children')['affairs'].agg(['count','mean','median','std'])

# Difference in means
children_yes = df[df['children']=='yes']['affairs']
children_no = df[df['children']=='no']['affairs']

t_stat, p_val = stats.ttest_ind(children_yes, children_no, equal_var=False)

# Effect size (Cohen's d, using pooled std with unequal sizes)
mean_diff = children_yes.mean() - children_no.mean()
var_yes = children_yes.var(ddof=1)
var_no = children_no.var(ddof=1)
pooled_sd = np.sqrt(((len(children_yes)-1)*var_yes + (len(children_no)-1)*var_no) / (len(children_yes)+len(children_no)-2))
cohen_d = mean_diff / pooled_sd if pooled_sd > 0 else np.nan

# Any-affair indicator
_df = df.copy()
_df['has_affair'] = (_df['affairs'] > 0).astype(int)

# Logistic regression for any affair
logit_model = smf.logit('has_affair ~ C(children, Treatment(reference="no")) + age + yearsmarried + religiousness + education + occupation + rating + C(gender)', data=_df).fit(disp=False)

# OLS for affairs frequency
ols_model = smf.ols('affairs ~ C(children, Treatment(reference="no")) + age + yearsmarried + religiousness + education + occupation + rating + C(gender)', data=_df).fit()

# Extract coefficients for children yes
logit_coef = logit_model.params.get('C(children, Treatment(reference="no"))[T.yes]', np.nan)
logit_p = logit_model.pvalues.get('C(children, Treatment(reference="no"))[T.yes]', np.nan)
ols_coef = ols_model.params.get('C(children, Treatment(reference="no"))[T.yes]', np.nan)
ols_p = ols_model.pvalues.get('C(children, Treatment(reference="no"))[T.yes]', np.nan)

# Compute proportion with any affairs by children
prop_any = _df.groupby('children')['has_affair'].mean()

# Save results for later inspection
result = {
    'summary': summary,
    'mean_diff_yes_minus_no': mean_diff,
    't_stat': t_stat,
    't_p': p_val,
    'cohen_d': cohen_d,
    'prop_any': prop_any,
    'logit_coef_yes': logit_coef,
    'logit_p_yes': logit_p,
    'ols_coef_yes': ols_coef,
    'ols_p_yes': ols_p,
}

print('SUMMARY\n', summary)
print('\nPROP_ANY\n', prop_any)
print('\nMEAN_DIFF_yes_minus_no', mean_diff)
print('T_TEST', t_stat, p_val)
print('COHEN_D', cohen_d)
print('\nLOGIT coef yes', logit_coef, 'p', logit_p)
print('OLS coef yes', ols_coef, 'p', ols_p)
