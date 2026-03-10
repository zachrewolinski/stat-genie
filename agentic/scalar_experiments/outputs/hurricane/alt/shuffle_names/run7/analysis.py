import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

pd.set_option('display.max_columns', 80)

# Load data

df = pd.read_csv('hurricane.csv')

# Rename for clarity (local only)
# Interpreted mapping based on ranges
# fatalities

df['fatalities'] = df['name']
# femininity ratings
# 'category' looks like 1-11 scale
# 'ind' looks like another 1-11 scale from MTurk

# severity proxies
# max wind speed
# min pressure
# saffir-simpson category (1-5)

# damage proxies
# elapsedyrs and source appear to be damage measures (large values)

# log transform for skewed variables

df['log_fatalities'] = np.log1p(df['fatalities'])

df['log_damage_2013'] = np.log1p(df['elapsedyrs'])
df['log_damage_2015'] = np.log1p(df['source'])

# Basic correlations
corrs = df[['fatalities','category','ind','masfem_mturk','gender_mf','year','ndam15','log_damage_2013','log_damage_2015','wind']].corr()
print('Correlation matrix (selected):')
print(corrs[['fatalities','category','ind','masfem_mturk']])

# Helper to fit and print model summaries (coeff and p-values)

def fit_model(formula, data):
    model = smf.ols(formula, data=data).fit(cov_type='HC3')
    print('\nModel:', formula)
    print(model.summary().tables[1])
    return model

# Models using category femininity rating
m1 = fit_model('log_fatalities ~ category', df)

m2 = fit_model('log_fatalities ~ category + year + ndam15 + gender_mf', df)

m3 = fit_model('log_fatalities ~ category + year + ndam15 + gender_mf + log_damage_2013 + wind', df)

# Alternative femininity measure from MTurk continuous rating
m4 = fit_model('log_fatalities ~ ind', df)

m5 = fit_model('log_fatalities ~ ind + year + ndam15 + gender_mf', df)

m6 = fit_model('log_fatalities ~ ind + year + ndam15 + gender_mf + log_damage_2013 + wind', df)

# Binary gender indicator
m7 = fit_model('log_fatalities ~ masfem_mturk', df)

m8 = fit_model('log_fatalities ~ masfem_mturk + year + ndam15 + gender_mf', df)

# Interaction between femininity and severity (max wind)
m9 = fit_model('log_fatalities ~ category * year + ndam15 + gender_mf', df)

# Negative binomial (count model) for fatalities using femininity rating
# Use fatalities as count; add small constant to avoid zeros in log link.

try:
    nb_model = smf.glm('fatalities ~ category + year + ndam15 + gender_mf', data=df,
                      family=sm.families.NegativeBinomial()).fit()
    print('\nNegative Binomial (fatalities) coeffs:')
    print(nb_model.summary().tables[1])
except Exception as e:
    print('Negative binomial failed:', e)

