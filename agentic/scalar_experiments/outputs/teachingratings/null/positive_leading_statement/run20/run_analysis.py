import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('teachingratings.csv')

# Basic stats
n = len(df)
beauty_mean = df['beauty'].mean()
beauty_std = df['beauty'].std()
eval_mean = df['eval'].mean()
eval_std = df['eval'].std()

# Correlation
corr, corr_p = stats.pearsonr(df['beauty'], df['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=df).fit(cov_type='HC3')

# Multiple OLS with controls
formula = (
    'eval ~ beauty + age + students + allstudents '
    '+ C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits)'
)
model_full = smf.ols(formula, data=df).fit(cov_type='HC3')

# Effect per 1 SD beauty
beauty_effect_simple = model_simple.params['beauty'] * beauty_std
beauty_effect_full = model_full.params['beauty'] * beauty_std

# R-squared
r2_simple = model_simple.rsquared
r2_full = model_full.rsquared

# Print results
print('N', n)
print('Beauty mean', beauty_mean, 'std', beauty_std)
print('Eval mean', eval_mean, 'std', eval_std)
print('Pearson r', corr, 'p', corr_p)

print('\nSimple OLS (HC3):')
print(model_simple.summary().tables[1])

print('\nFull OLS (HC3):')
print(model_full.summary().tables[1])

print('\nEffect per 1 SD beauty:')
print('Simple', beauty_effect_simple)
print('Full', beauty_effect_full)

print('\nR2 simple', r2_simple)
print('R2 full', r2_full)
