import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


df = pd.read_csv('mortgage.csv')
print('rows', len(df))
print('columns', df.columns.tolist())

# Ensure binary columns are numeric
for col in ['female', 'accept', 'deny']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Basic counts
print(df[['female','accept','deny']].describe(include='all'))

# Drop rows with missing key vars
key_cols = ['female','accept']
sub = df[key_cols].dropna()

# Contingency table
ct = pd.crosstab(sub['female'], sub['accept'])
print('contingency:\n', ct)

# Chi-square test
chi2, p, dof, exp = stats.chi2_contingency(ct)
print('chi2', chi2, 'p', p)

# Difference in proportions (female=1 vs male=0)
prop_f = sub.loc[sub['female']==1, 'accept'].mean()
prop_m = sub.loc[sub['female']==0, 'accept'].mean()
print('prop_f', prop_f, 'prop_m', prop_m, 'diff', prop_f-prop_m)

# Wald test for difference in proportions
n_f = (sub['female']==1).sum()
n_m = (sub['female']==0).sum()
# pooled variance for difference in proportions
p_pool = sub['accept'].mean()
se = np.sqrt(p_pool*(1-p_pool)*(1/n_f + 1/n_m))
if se>0:
    z = (prop_f-prop_m)/se
    p_diff = 2*(1-stats.norm.cdf(abs(z)))
    print('z', z, 'p_diff', p_diff)

# Logistic regression with controls
controls = ['black','housing_expense_ratio','self_employed','married','mortgage_credit',
            'consumer_credit','bad_history','PI_ratio','loan_to_value','denied_PMI']

cols = ['accept','female'] + [c for c in controls if c in df.columns]
reg = df[cols].dropna()
print('reg rows', len(reg))

X = reg.drop(columns=['accept'])
X = sm.add_constant(X, has_constant='add')
model = sm.Logit(reg['accept'], X).fit(disp=False)
print(model.summary())

# odds ratio for female
coef = model.params['female']
se = model.bse['female']
or_val = np.exp(coef)
ci_low = np.exp(coef - 1.96*se)
ci_high = np.exp(coef + 1.96*se)
print('female coef', coef, 'se', se, 'OR', or_val, 'CI', (ci_low, ci_high), 'p', model.pvalues['female'])
