import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('reading.csv')

# Identify dyslexic participants
if 'dyslexia_bin' in df.columns:
    dys = df[df['dyslexia_bin'] == 1].copy()
else:
    dys = df[df['dyslexia'].isin([1,2])].copy()

# log speed

dys = dys[dys['speed'] > 0].copy()

dys['log_speed'] = np.log(dys['speed'])

# OLS with page fixed effects and num_words

formula = 'log_speed ~ reader_view + C(page_id) + num_words'

model = smf.ols(formula, data=dys).fit()

coef = model.params['reader_view']
se = model.bse['reader_view']
p = model.pvalues['reader_view']

# percent change approx
pct = (np.exp(coef) - 1) * 100

print('OLS reader_view coef (log speed):', coef)
print('SE:', se)
print('p:', p)
print('Approx percent change:', pct)

