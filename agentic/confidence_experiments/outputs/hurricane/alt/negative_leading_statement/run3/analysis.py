import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('hurricane.csv')

# Basic sanity
print(df.head())
print(df.describe(include='all').transpose())

# Create transformed outcomes
# log1p deaths to reduce skew

df['log_alldeaths'] = np.log1p(df['alldeaths'])

# Standardize some controls for comparability (optional)

# Key variables

# Simple correlations
print('\nCorrelation masfem vs deaths (log1p):', df['masfem'].corr(df['log_alldeaths']))
print('Correlation masfem_mturk vs deaths (log1p):', df['masfem_mturk'].corr(df['log_alldeaths']))
print('Correlation gender_mf vs deaths (log1p):', df['gender_mf'].corr(df['log_alldeaths']))

# OLS models
# Model 1: log deaths ~ masfem
m1 = smf.ols('log_alldeaths ~ masfem', data=df).fit(cov_type='HC3')
print('\nModel 1: log_alldeaths ~ masfem (HC3)')
print(m1.summary())

# Model 2: log deaths ~ masfem + category + wind + min
m2 = smf.ols('log_alldeaths ~ masfem + category + wind + min', data=df).fit(cov_type='HC3')
print('\nModel 2: log_alldeaths ~ masfem + category + wind + min (HC3)')
print(m2.summary())

# Model 3: log deaths ~ masfem + category + wind + min + ndam15
m3 = smf.ols('log_alldeaths ~ masfem + category + wind + min + ndam15', data=df).fit(cov_type='HC3')
print('\nModel 3: log_alldeaths ~ masfem + category + wind + min + ndam15 (HC3)')
print(m3.summary())

# Model 4: use masfem_mturk instead
m4 = smf.ols('log_alldeaths ~ masfem_mturk + category + wind + min', data=df).fit(cov_type='HC3')
print('\nModel 4: log_alldeaths ~ masfem_mturk + category + wind + min (HC3)')
print(m4.summary())

# Model 5: binary gender indicator
m5 = smf.ols('log_alldeaths ~ gender_mf + category + wind + min', data=df).fit(cov_type='HC3')
print('\nModel 5: log_alldeaths ~ gender_mf + category + wind + min (HC3)')
print(m5.summary())

# Alternative outcome: ndam15 (log)

df['log_ndam15'] = np.log1p(df['ndam15'])

m6 = smf.ols('log_ndam15 ~ masfem + category + wind + min', data=df).fit(cov_type='HC3')
print('\nModel 6: log_ndam15 ~ masfem + category + wind + min (HC3)')
print(m6.summary())

# Poisson regression on deaths (counts)
# Guard against zeros

try:
    m7 = smf.glm('alldeaths ~ masfem + category + wind + min', data=df, family=sm.families.Poisson()).fit(cov_type='HC3')
    print('\nModel 7: Poisson alldeaths ~ masfem + category + wind + min (HC3)')
    print(m7.summary())
except Exception as e:
    print('Poisson model failed:', e)

# Save key results to file for reference

results = {
    'm1': m1.params.to_dict(),
    'm1_p': m1.pvalues.to_dict(),
    'm2': m2.params.to_dict(),
    'm2_p': m2.pvalues.to_dict(),
    'm3': m3.params.to_dict(),
    'm3_p': m3.pvalues.to_dict(),
    'm4': m4.params.to_dict(),
    'm4_p': m4.pvalues.to_dict(),
    'm5': m5.params.to_dict(),
    'm5_p': m5.pvalues.to_dict(),
    'm6': m6.params.to_dict(),
    'm6_p': m6.pvalues.to_dict(),
}

pd.Series({k: v for k, v in results.items()}).to_json('analysis_results.json')

