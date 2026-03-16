import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

print('rows', _df.shape)
print('columns', _df.columns.tolist())

# Basic missingness
print('\nMissing values per column (nonzero):')
missing = _df.isna().sum()
print(missing[missing>0])

# Create log deaths
_df['log_deaths'] = np.log1p(_df['alldeaths'])

# Basic correlations
for col in ['masfem', 'masfem_mturk', 'gender_mf']:
    if col in _df.columns:
        corr = _df[[col, 'alldeaths']].corr().iloc[0,1]
        corr_log = _df[[col, 'log_deaths']].corr().iloc[0,1]
        print(f'corr {col} vs alldeaths: {corr:.3f}, vs log_deaths: {corr_log:.3f}')

# Simple OLS: log_deaths ~ masfem
print('\nOLS: log_deaths ~ masfem')
model1 = smf.ols('log_deaths ~ masfem', data=_df).fit()
print(model1.summary().tables[1])

# OLS with controls (severity): wind, min, category, ndam15 (damage), year
# Use available columns and drop rows with NA
controls = ['wind', 'min', 'category', 'ndam15', 'year']
for ctrl in controls:
    if ctrl not in _df.columns:
        controls.remove(ctrl)

formula = 'log_deaths ~ masfem'
for ctrl in controls:
    formula += f' + {ctrl}'
print('\nOLS with controls:', formula)
model2 = smf.ols(formula, data=_df).fit()
print(model2.summary().tables[1])

# Alternative: using gender_mf instead of masfem
if 'gender_mf' in _df.columns:
    formula_g = 'log_deaths ~ gender_mf'
    for ctrl in controls:
        formula_g += f' + {ctrl}'
    print('\nOLS with controls (gender_mf):', formula_g)
    model3 = smf.ols(formula_g, data=_df).fit()
    print(model3.summary().tables[1])

# Negative binomial / Poisson? Try Poisson on deaths
print('\nPoisson: alldeaths ~ masfem + controls')
try:
    poisson = smf.glm('alldeaths ~ masfem' + ''.join([f' + {c}' for c in controls]), data=_df, family=sm.families.Poisson()).fit()
    print(poisson.summary().tables[1])
except Exception as e:
    print('Poisson failed:', e)

