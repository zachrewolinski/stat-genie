import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('hurricane.csv')

# Select relevant columns
cols = ['alldeaths', 'masfem', 'gender_mf', 'category', 'wind', 'min', 'year']
d = df[cols].copy()

# Check missing
print('Missing values:')
print(d.isna().sum())

# Transform outcome

d['log_deaths'] = np.log1p(d['alldeaths'])

# Correlations
print('\nPearson correlations with log_deaths:')
for col in ['masfem', 'gender_mf', 'category', 'wind', 'min', 'year']:
    r, p = stats.pearsonr(d[col], d['log_deaths'])
    print(f'{col}: r={r:.3f}, p={p:.4f}')

print('\nSpearman correlations with log_deaths:')
for col in ['masfem', 'gender_mf', 'category', 'wind', 'min', 'year']:
    r, p = stats.spearmanr(d[col], d['log_deaths'])
    print(f'{col}: rho={r:.3f}, p={p:.4f}')

# OLS with controls
model1 = smf.ols('log_deaths ~ masfem + category + wind + min + year', data=d).fit(cov_type='HC3')
print('\nOLS log_deaths ~ masfem + controls (HC3):')
print(model1.summary().tables[1])

model2 = smf.ols('log_deaths ~ gender_mf + category + wind + min + year', data=d).fit(cov_type='HC3')
print('\nOLS log_deaths ~ gender_mf + controls (HC3):')
print(model2.summary().tables[1])

# Poisson and Negative Binomial

pois1 = smf.glm('alldeaths ~ masfem + category + wind + min + year', data=d, family=sm.families.Poisson()).fit(cov_type='HC3')
print('\nPoisson alldeaths ~ masfem + controls (HC3):')
print(pois1.summary().tables[1])

nb1 = smf.glm('alldeaths ~ masfem + category + wind + min + year', data=d, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
print('\nNegBin alldeaths ~ masfem + controls (HC3):')
print(nb1.summary().tables[1])

# Gender binary in GLM
pois2 = smf.glm('alldeaths ~ gender_mf + category + wind + min + year', data=d, family=sm.families.Poisson()).fit(cov_type='HC3')
print('\nPoisson alldeaths ~ gender_mf + controls (HC3):')
print(pois2.summary().tables[1])

nb2 = smf.glm('alldeaths ~ gender_mf + category + wind + min + year', data=d, family=sm.families.NegativeBinomial()).fit(cov_type='HC3')
print('\nNegBin alldeaths ~ gender_mf + controls (HC3):')
print(nb2.summary().tables[1])

