import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Basic info
print('columns:', list(_df.columns))
print('n_rows:', len(_df))
print(_df.head())

# Identify key variables
# feature4: femininity index; feature6: binary female
# feature8: deaths; feature9/feature14: damage; feature7: category; feature5: pressure; feature13: wind; feature2: year

# Clean: ensure numeric
numeric_cols = [c for c in _df.columns if c.startswith('feature') and _df[c].dtype != 'O']
for c in numeric_cols:
    _df[c] = pd.to_numeric(_df[c], errors='coerce')

# Create log-transformed outcomes
_df['log_deaths'] = np.log1p(_df['feature8'])
_df['log_damage_2013'] = np.log1p(_df['feature9'])
_df['log_damage_2015'] = np.log1p(_df['feature14'])

# Correlations
corrs = _df[['feature4','feature6','feature8','feature9','feature14','feature7','feature5','feature13']].corr()
print('\ncorrelations:\n', corrs)

# OLS models
# Base: log deaths ~ femininity index
m1 = smf.ols('log_deaths ~ feature4', data=_df).fit()
# Add intensity controls
m2 = smf.ols('log_deaths ~ feature4 + feature7 + feature5 + feature13 + feature2', data=_df).fit()

# Use female indicator
m3 = smf.ols('log_deaths ~ feature6', data=_df).fit()
m4 = smf.ols('log_deaths ~ feature6 + feature7 + feature5 + feature13 + feature2', data=_df).fit()

# Damage models
m5 = smf.ols('log_damage_2013 ~ feature4', data=_df).fit()
m6 = smf.ols('log_damage_2013 ~ feature4 + feature7 + feature5 + feature13 + feature2', data=_df).fit()

# Print summaries (coef, p)
models = {'m1': m1, 'm2': m2, 'm3': m3, 'm4': m4, 'm5': m5, 'm6': m6}
for name, m in models.items():
    print(f"\n{name}:")
    print(m.params)
    print(m.pvalues)
    print('r2', m.rsquared)

# Also Poisson for deaths (counts), with log link and robust SE
try:
    m7 = smf.glm('feature8 ~ feature4 + feature7 + feature5 + feature13 + feature2', data=_df, family=sm.families.Poisson()).fit(cov_type='HC3')
    print('\nPoisson deaths (feature4):')
    print(m7.params)
    print(m7.pvalues)
except Exception as e:
    print('Poisson failed', e)

# Output key stats for interpretation
print('\nfeature4 range', _df['feature4'].min(), _df['feature4'].max())
print('feature6 counts', _df['feature6'].value_counts(dropna=False))
print('deaths summary', _df['feature8'].describe())
