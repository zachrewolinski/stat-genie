import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('mortgage.csv')

# Define outcome and predictors
outcome = 'accept'
predictors = [
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
    'denied_PMI',
]

# Drop rows with missing values in required columns
cols = [outcome] + predictors
_df = _df[cols].dropna().copy()

# Descriptive acceptance rates by gender
accept_rates = _df.groupby('female')[outcome].mean()

# Logistic regression: accept ~ female + controls
X = sm.add_constant(_df[predictors])
model = sm.Logit(_df[outcome], X).fit(disp=False)
params = model.params
pvalues = model.pvalues

female_coef = params['female']
female_p = pvalues['female']
female_or = float(np.exp(female_coef))

print('Rows used:', len(_df))
print('Acceptance rate by female (0=male,1=female):')
print(accept_rates)
print('\nLogit coefficient for female:')
print('coef=', female_coef)
print('p-value=', female_p)
print('odds_ratio=', female_or)
