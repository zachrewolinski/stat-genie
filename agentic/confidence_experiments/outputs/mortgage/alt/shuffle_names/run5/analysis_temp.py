import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('mortgage.csv')
print('shape', df.shape)
print('deny mean', df['deny'].mean())
print('accept mean', df['accept'].mean())
# Check relationship between accept and deny
print('accept equals 1-deny', np.mean(df['accept'] == (1 - df['deny'])))
print('accept equals deny', np.mean(df['accept'] == df['deny']))
print('corr accept/deny', df['accept'].corr(df['deny']))

# frequency table
for col in ['deny','accept','female']:
    print(col, df[col].value_counts().to_dict())

# Crosstab for deny vs female
ct_deny = pd.crosstab(df['female'], df['deny'])
print('crosstab female x deny')
print(ct_deny)

# If deny==1 is denial, then approval=1-deny
# We'll compute approval rates by gender under both interpretations

df['approve_if_deny1_is_denial'] = 1 - df['deny']

df['approve_if_deny1_is_accept'] = df['deny']

for label, col in [('deny1_is_denial','approve_if_deny1_is_denial'),
                   ('deny1_is_accept','approve_if_deny1_is_accept')]:
    rates = df.groupby('female')[col].mean()
    print(label, rates.to_dict())

# logistic regression for approval using approve_if_deny1_is_denial
# We'll use a basic model with female only, then with controls

df['approve'] = df['approve_if_deny1_is_denial']

# model 1: approve ~ female
m1 = smf.logit('approve ~ female', data=df).fit(disp=False)
print('m1', m1.params, m1.pvalues)

# model 2: add controls (all other columns except deny/accept?)
# Choose plausible controls: bad_history, denied_PMI, consumer_credit, mortgage_credit,
# loan_to_value, married, black, PI_ratio, housing_expense_ratio, self_employed
controls = ['bad_history','denied_PMI','consumer_credit','mortgage_credit',
            'loan_to_value','married','black','PI_ratio','housing_expense_ratio','self_employed','Unnamed: 0']
formula = 'approve ~ female + ' + ' + '.join(controls)

m2 = smf.logit(formula, data=df).fit(disp=False)
print('m2 female coef', m2.params['female'], 'p', m2.pvalues['female'])

# compute marginal effect of female in m2 at mean
margeff = m2.get_margeff(at='mean', method='dydx')
print(margeff.summary())
