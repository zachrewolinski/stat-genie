import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Basic cleaning
# Ensure numeric columns
for col in ['masfem','masfem_mturk','gender_mf','alldeaths','wind','min','category','ndam','ndam15','year']:
    if col in _df.columns:
        _df[col] = pd.to_numeric(_df[col], errors='coerce')

# Create log outcomes to handle skew
_df['log_deaths'] = np.log1p(_df['alldeaths'])
_df['log_ndam15'] = np.log1p(_df['ndam15'])
_df['log_ndam'] = np.log1p(_df['ndam'])

# Drop rows with missing key vars
base_cols = ['masfem','gender_mf','wind','min','category','year','log_deaths']
_df_base = _df.dropna(subset=base_cols)

# Regression models
models = {}

# Model 1: log_deaths ~ masfem + controls
models['m1_masfem'] = smf.ols('log_deaths ~ masfem + wind + min + category + year', data=_df_base).fit(cov_type='HC3')

# Model 2: log_deaths ~ gender_mf + controls
models['m2_gender'] = smf.ols('log_deaths ~ gender_mf + wind + min + category + year', data=_df_base).fit(cov_type='HC3')

# Model 3: interaction masfem * wind
models['m3_masfem_int'] = smf.ols('log_deaths ~ masfem*wind + min + category + year', data=_df_base).fit(cov_type='HC3')

# Damage outcome as proxy
_df_damage = _df.dropna(subset=['masfem','wind','min','category','year','log_ndam15'])
models['m4_damage'] = smf.ols('log_ndam15 ~ masfem + wind + min + category + year', data=_df_damage).fit(cov_type='HC3')

# Print key results
print('N base:', len(_df_base))
print('N damage:', len(_df_damage))

for name, model in models.items():
    print('\n', name)
    print(model.summary().tables[1])

# Simple correlations
corr = _df[['masfem','alldeaths','wind','min','category']].corr()
print('\nCorrelation matrix (subset):')
print(corr)
