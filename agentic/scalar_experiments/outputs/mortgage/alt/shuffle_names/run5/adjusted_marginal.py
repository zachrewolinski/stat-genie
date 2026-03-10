import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('mortgage.csv')

female_col = 'denied_PMI'
approve_col = 'deny'

controls = [
    'consumer_credit',
    'mortgage_credit',
    'accept',
    'loan_to_value',
    'married',
    'black',
    'PI_ratio',
    'housing_expense_ratio',
    'female',
    'Unnamed: 0',
]

formula = f"{approve_col} ~ {female_col} + " + " + ".join(controls)

m2 = smf.logit(formula, data=df).fit(disp=False)

margeff = m2.get_margeff(at='overall', method='dydx')
print(margeff.summary())
