import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

_df = pd.read_csv('mortgage.csv')

# Define outcome
if 'accept' in _df.columns:
    outcome = _df['accept']
elif 'deny' in _df.columns:
    outcome = 1 - _df['deny']
else:
    raise ValueError('No accept or deny column found')

# Basic stats (use full data for raw rates where possible)
counts = _df.groupby('female')[outcome.name].agg(['count', 'sum'])

count_male = counts.loc[0, 'sum']
count_female = counts.loc[1, 'sum']
n_male = counts.loc[0, 'count']
n_female = counts.loc[1, 'count']

p_male = count_male / n_male
p_female = count_female / n_female

p_pool = (count_male + count_female) / (n_male + n_female)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_male + 1/n_female))
if se == 0:
    z_stat = np.nan
    p_value = np.nan
else:
    z_stat = (p_female - p_male) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

# Logistic regression with controls
control_cols = [
    'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]

X = _df[control_cols].copy()
Y = outcome.copy()

# Remove inf and missing
X = X.replace([np.inf, -np.inf], np.nan)
Y = Y.replace([np.inf, -np.inf], np.nan)

mask = X.notna().all(axis=1) & Y.notna()
X = X.loc[mask]
Y = Y.loc[mask]

X = sm.add_constant(X)

model = sm.GLM(Y, X, family=sm.families.Binomial())
res = model.fit(cov_type='HC1')

female_coef = res.params['female']
female_se = res.bse['female']
female_p = res.pvalues['female']

margeff = res.get_margeff(at='overall')
me_table = margeff.summary_frame()

female_me = me_table.loc['female', 'dy/dx']
female_me_se = me_table.loc['female', 'Std. Err.']
female_me_p = me_table.loc['female', 'Pr(>|z|)']

print('N_total:', len(_df))
print('N_model:', len(Y))
print('Approval rate male:', p_male)
print('Approval rate female:', p_female)
print('Difference (female - male):', p_female - p_male)
print('Two-proportion z-test z:', z_stat, 'p:', p_value)
print('Logit coef female:', female_coef, 'SE:', female_se, 'p:', female_p)
print('AME female:', female_me, 'SE:', female_me_se, 'p:', female_me_p)
