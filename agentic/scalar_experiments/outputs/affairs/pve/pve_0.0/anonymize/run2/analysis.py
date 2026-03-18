import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('affairs.csv')

# Ensure expected columns
cols = _df.columns.tolist()

# Map children variable
# feature6: yes/no
_df['children_yes'] = _df['feature6'].map({'yes': 1, 'no': 0})

# Outcome: affair frequency
_df['affair_freq'] = _df['feature2']

# Indicator for any affair
_df['affair_any'] = (_df['affair_freq'] > 0).astype(int)

# Group stats
grp = _df.groupby('children_yes')['affair_freq']
summary = grp.agg(['count','mean','median','std'])

# Welch t-test
no = _df[_df['children_yes'] == 0]['affair_freq']
yes = _df[_df['children_yes'] == 1]['affair_freq']

t_stat, t_p = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')

# Mann-Whitney U (two-sided)
# Use alternative='two-sided' for scipy>=1.7
try:
    u_stat, u_p = stats.mannwhitneyu(yes, no, alternative='two-sided')
except TypeError:
    u_stat, u_p = stats.mannwhitneyu(yes, no)

# Cohen's d
# Pooled SD for two groups
n1, n2 = len(yes), len(no)
mean1, mean2 = yes.mean(), no.mean()
var1, var2 = yes.var(ddof=1), no.var(ddof=1)
pooled_sd = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
cohens_d = (mean1 - mean2) / pooled_sd if pooled_sd > 0 else np.nan

# Logistic regression for affair_any ~ children_yes
logit_model = smf.logit('affair_any ~ children_yes', data=_df).fit(disp=False)
logit_params = logit_model.params
logit_pvalues = logit_model.pvalues
odds_ratio = np.exp(logit_params['children_yes'])

# Linear regression with controls (optional)
# Controls: age (feature4), years married (feature5), religiosity (feature7), education (feature8), occupation (feature9), marriage rating (feature10), gender (feature3)
# Use robust SE (HC3)
ols_formula = 'affair_freq ~ children_yes + feature4 + feature5 + feature7 + feature8 + feature9 + feature10 + C(feature3)'
ols_model = smf.ols(ols_formula, data=_df).fit(cov_type='HC3')

# Collect results
results = {
    'columns': cols,
    'summary': summary.to_dict(),
    't_test': {'t_stat': t_stat, 'p_value': t_p},
    'mannwhitney': {'u_stat': u_stat, 'p_value': u_p},
    'cohens_d': cohens_d,
    'logit': {
        'coef_children': logit_params['children_yes'],
        'p_value_children': logit_pvalues['children_yes'],
        'odds_ratio_children': odds_ratio
    },
    'ols': {
        'coef_children': ols_model.params['children_yes'],
        'p_value_children': ols_model.pvalues['children_yes']
    }
}

# Print concise results
print('Group summary (affair_freq) by children_yes:')
print(summary)
print('\nWelch t-test p-value:', t_p)
print('Mann-Whitney p-value:', u_p)
print('Cohen d (yes - no):', cohens_d)
print('\nLogit (affair_any ~ children_yes): coef, OR, p-value')
print(logit_params['children_yes'], odds_ratio, logit_pvalues['children_yes'])
print('\nOLS with controls (affair_freq ~ children_yes + controls): coef, p-value')
print(ols_model.params['children_yes'], ols_model.pvalues['children_yes'])
