import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind

# Load data
_df = pd.read_csv('affairs.csv')

# Basic checks
_df['any_affair'] = (_df['affairs'] > 0).astype(int)

# Group summaries
summary = _df.groupby('children').agg(
    n=('affairs', 'size'),
    mean_affairs=('affairs', 'mean'),
    median_affairs=('affairs', 'median'),
    prop_any_affair=('any_affair', 'mean')
)

# Two-sample t-test for mean affairs (unequal variances)
with_children = _df[_df['children'] == 'yes']['affairs']
without_children = _df[_df['children'] == 'no']['affairs']

t_stat, p_val, _ = ttest_ind(with_children, without_children, usevar='unequal')

# OLS on count with controls (robust SE)
ols_formula = (
    'affairs ~ C(children) + C(gender) + age + yearsmarried + '
    'religiousness + education + occupation + rating'
)
ols_model = smf.ols(ols_formula, data=_df).fit(cov_type='HC3')

# Logistic regression for any affair
logit_formula = (
    'any_affair ~ C(children) + C(gender) + age + yearsmarried + '
    'religiousness + education + occupation + rating'
)
logit_model = smf.logit(logit_formula, data=_df).fit(disp=0)

# Extract key coefficients
ols_coef = ols_model.params.get('C(children)[T.yes]', np.nan)
ols_p = ols_model.pvalues.get('C(children)[T.yes]', np.nan)

logit_coef = logit_model.params.get('C(children)[T.yes]', np.nan)
logit_p = logit_model.pvalues.get('C(children)[T.yes]', np.nan)

# Print results
print('Group summary by children:')
print(summary)
print('\nT-test (affairs mean, children yes vs no):')
print({'t_stat': t_stat, 'p_value': p_val})

print('\nOLS (affairs count) coefficient for children=yes:')
print({'coef': ols_coef, 'p_value': ols_p})

print('\nLogit (any affair) coefficient for children=yes:')
print({'coef': logit_coef, 'p_value': logit_p})

# Save key stats for conclusion use
results = {
    'mean_affairs_yes': summary.loc['yes', 'mean_affairs'],
    'mean_affairs_no': summary.loc['no', 'mean_affairs'],
    'prop_any_affair_yes': summary.loc['yes', 'prop_any_affair'],
    'prop_any_affair_no': summary.loc['no', 'prop_any_affair'],
    'ttest_p': p_val,
    'ols_coef_children_yes': ols_coef,
    'ols_p_children_yes': ols_p,
    'logit_coef_children_yes': logit_coef,
    'logit_p_children_yes': logit_p,
}

pd.Series(results).to_csv('analysis_results.csv')
