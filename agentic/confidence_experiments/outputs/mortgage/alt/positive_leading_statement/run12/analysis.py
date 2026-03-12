import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data

df = pd.read_csv('mortgage.csv')

# Basic checks
print('rows', len(df))
print(df[['female','accept','deny']].head())

# Approval rate by gender
rate_by_gender = df.groupby('female')['accept'].mean()
count_by_gender = df.groupby('female')['accept'].agg(['count','sum'])
print('approval rate by gender:\n', rate_by_gender)
print('counts by gender:\n', count_by_gender)

# 2x2 chi-square test on accept vs female
contingency = pd.crosstab(df['female'], df['accept'])
print('contingency\n', contingency)
chi2, p, dof, exp = stats.chi2_contingency(contingency)
print('chi2', chi2, 'p', p)

# Logistic regression: accept ~ female (unadjusted)
model_unadj = smf.logit('accept ~ female', data=df).fit(disp=False)
print(model_unadj.summary())

# Logistic regression with controls (creditworthiness variables)
# We'll include key variables from dataset; avoid multicollinearity using both deny/accept; choose accept as outcome
controls = [
    'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]
formula = 'accept ~ female + ' + ' + '.join(controls)
model_adj = smf.logit(formula, data=df).fit(disp=False)
print(model_adj.summary())

# Odds ratio for female in adjusted model
params = model_adj.params
conf = model_adj.conf_int()
odds_ratio = np.exp(params['female'])
conf_or = np.exp(conf.loc['female'])
print('female OR', odds_ratio, 'CI', conf_or.tolist())

# Also compute marginal effect of female (average marginal effect)
me = model_adj.get_margeff(at='overall', method='dydx')
print(me.summary())

