import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('affairs.csv')

# Basic summaries
summary = df.groupby('children')['affairs'].agg(['count','mean','median','std'])
summary['affairs_gt0'] = df.groupby('children')['affairs'].apply(lambda s: (s>0).mean())

print('Summary by children:')
print(summary)
print('\nOverall affairs mean:', df['affairs'].mean())

# Two-sample t-test for mean difference
from scipy import stats

g_yes = df[df['children']=='yes']['affairs']
g_no = df[df['children']=='no']['affairs']

# Welch t-test
t_stat, p_val = stats.ttest_ind(g_yes, g_no, equal_var=False)
print('\nWelch t-test (children yes vs no) t=%.4f p=%.6f' % (t_stat, p_val))
print('Mean yes:', g_yes.mean(), 'Mean no:', g_no.mean())

# Difference in proportion with any affair
prop_yes = (g_yes>0).mean()
prop_no = (g_no>0).mean()
print('\nProportion affairs>0 yes:', prop_yes, 'no:', prop_no, 'diff yes-no:', prop_yes-prop_no)

# Logistic regression for affairs>0

df['affairs_gt0'] = (df['affairs']>0).astype(int)
logit = smf.logit('affairs_gt0 ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating', data=df).fit(disp=0)
print('\nLogit regression (affairs>0):')
print(logit.summary().tables[1])

# OLS on log(affairs+1)
df['log_affairs'] = np.log1p(df['affairs'])
ols = smf.ols('log_affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating', data=df).fit(cov_type='HC3')
print('\nOLS log(affairs+1) with robust SE:')
print(ols.summary().tables[1])

# Poisson regression for counts
poisson = smf.glm('affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating', data=df, family=sm.families.Poisson()).fit(cov_type='HC3')
print('\nPoisson regression:')
print(poisson.summary().tables[1])

# Save key results for later
results = {
    'mean_yes': g_yes.mean(),
    'mean_no': g_no.mean(),
    'prop_yes': prop_yes,
    'prop_no': prop_no,
    'ttest_p': p_val,
    'logit_children_coef': logit.params.get('C(children)[T.yes]', np.nan),
    'logit_children_p': logit.pvalues.get('C(children)[T.yes]', np.nan),
    'ols_children_coef': ols.params.get('C(children)[T.yes]', np.nan),
    'ols_children_p': ols.pvalues.get('C(children)[T.yes]', np.nan),
    'poisson_children_coef': poisson.params.get('C(children)[T.yes]', np.nan),
    'poisson_children_p': poisson.pvalues.get('C(children)[T.yes]', np.nan)
}

print('\nKey results:')
for k, v in results.items():
    print(k, v)
