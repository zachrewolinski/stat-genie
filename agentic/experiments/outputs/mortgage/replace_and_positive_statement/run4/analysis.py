import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv('mortgage.csv')

# Create a binary gender indicator if female column isn't already 0/1
# Using 0.5 threshold aligns with the mid-point of a 0/1 variable and keeps balance
female_binary = (df['female'] >= 0.5).astype(int)

df = df.copy()
df['female_binary'] = female_binary

# Outcome: accept (1 accepted, 0 denied)
# Note: accept/deny are not perfect complements in this file; use accept as provided.

# Simple acceptance rates by gender
accept_rate_by_gender = df.groupby('female_binary')['accept'].mean()

# Build logistic regression with controls
predictors = [
    'female_binary',
    'black',
    'housing_expense_ratio',
    'self_employed',
    'married',
    'mortgage_credit',
    'consumer_credit',
    'bad_history',
    'PI_ratio',
    'loan_to_value',
    'denied_PMI'
]

X = df[predictors]
X = sm.add_constant(X)
y = df['accept']

logit_model = sm.Logit(y, X, missing='drop')
logit_res = logit_model.fit(disp=False)

# Also fit a model using the raw female variable (if someone prefers continuous signal)
X_cont = df[[
    'female',
    'black',
    'housing_expense_ratio',
    'self_employed',
    'married',
    'mortgage_credit',
    'consumer_credit',
    'bad_history',
    'PI_ratio',
    'loan_to_value',
    'denied_PMI'
]]
X_cont = sm.add_constant(X_cont)
logit_model_cont = sm.Logit(y, X_cont, missing='drop')
logit_res_cont = logit_model_cont.fit(disp=False)

# Collect key results
results = {
    'accept_rate_by_gender': accept_rate_by_gender,
    'female_binary_coef': logit_res.params['female_binary'],
    'female_binary_pvalue': logit_res.pvalues['female_binary'],
    'female_coef_cont': logit_res_cont.params['female'],
    'female_pvalue_cont': logit_res_cont.pvalues['female'],
}

print('Acceptance rate by female_binary (0=male,1=female):')
print(accept_rate_by_gender)
print('\nLogit w/ female_binary: coef, p-value')
print(results['female_binary_coef'], results['female_binary_pvalue'])
print('\nLogit w/ female (continuous): coef, p-value')
print(results['female_coef_cont'], results['female_pvalue_cont'])
