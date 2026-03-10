import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load

df = pd.read_csv('hurricane.csv')

# Prepare design matrix

formula = 'alldeaths ~ masfem + wind + min + category'

# Use discrete NegativeBinomial (NB2) to estimate alpha
nb2 = smf.negativebinomial(formula, data=df).fit(disp=False)
print(nb2.summary().tables[1])

# with masfem_mturk
nb2_mturk = smf.negativebinomial('alldeaths ~ masfem_mturk + wind + min + category', data=df).fit(disp=False)
print('\nNB2 mturk')
print(nb2_mturk.summary().tables[1])

# with gender
nb2_gender = smf.negativebinomial('alldeaths ~ gender_mf + wind + min + category', data=df).fit(disp=False)
print('\nNB2 gender')
print(nb2_gender.summary().tables[1])

# Check alpha estimates
print('\nalpha', nb2.params.get('alpha'))
