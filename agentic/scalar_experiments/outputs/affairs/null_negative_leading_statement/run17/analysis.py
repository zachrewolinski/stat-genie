import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('affairs.csv')

# Basic grouping
summary = df.groupby('children').agg(
    n=('affairs', 'size'),
    mean_affairs=('affairs', 'mean'),
    median_affairs=('affairs', 'median'),
    prop_any_affair=('affairs', lambda x: (x > 0).mean()),
)

# Two-sample t-test on affairs counts (unequal variances)
children_yes = df.loc[df['children'] == 'yes', 'affairs']
children_no = df.loc[df['children'] == 'no', 'affairs']

# Handle potential empty groups (shouldn't happen)
if len(children_yes) > 1 and len(children_no) > 1:
    t_res = stats.ttest_ind(children_yes, children_no, equal_var=False)
else:
    t_res = None

# Logistic regression for any affair

df['any_affair'] = (df['affairs'] > 0).astype(int)

# Use categorical for gender, children
logit_model = smf.logit(
    'any_affair ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating',
    data=df,
).fit(disp=False)

# Negative binomial regression for counts
nb_model = smf.glm(
    'affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating',
    data=df,
    family=sm.families.NegativeBinomial(),
).fit()

# Extract children coefficient (yes vs no), using no as baseline (alphabetical)
# statsmodels uses first category as baseline by default; check parameter name
child_param_logit = [p for p in logit_model.params.index if p.startswith('C(children)')]
child_param_nb = [p for p in nb_model.params.index if p.startswith('C(children)')]

results = {
    'summary': summary,
    't_test': None if t_res is None else {
        't_stat': float(t_res.statistic),
        'p_value': float(t_res.pvalue),
        'mean_yes': float(children_yes.mean()),
        'mean_no': float(children_no.mean()),
    },
    'logit_children_params': {p: {
        'coef': float(logit_model.params[p]),
        'p_value': float(logit_model.pvalues[p]),
        'odds_ratio': float(np.exp(logit_model.params[p])),
    } for p in child_param_logit},
    'nb_children_params': {p: {
        'coef': float(nb_model.params[p]),
        'p_value': float(nb_model.pvalues[p]),
        'rate_ratio': float(np.exp(nb_model.params[p])),
    } for p in child_param_nb},
}

print('SUMMARY')
print(summary.to_string())
print('\nT_TEST')
print(results['t_test'])
print('\nLOGIT_CHILDREN')
print(results['logit_children_params'])
print('\nNB_CHILDREN')
print(results['nb_children_params'])
