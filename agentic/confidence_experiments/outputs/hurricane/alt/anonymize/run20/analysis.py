import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Rename for clarity
cols = {
    'feature1': 'id',
    'feature2': 'year',
    'feature3': 'name',
    'feature4': 'masfem_coders',
    'feature5': 'min_pressure',
    'feature6': 'female_binary',
    'feature7': 'category',
    'feature8': 'deaths',
    'feature9': 'damage_2013',
    'feature10': 'years_elapsed',
    'feature11': 'source',
    'feature12': 'masfem_mturk',
    'feature13': 'max_wind',
    'feature14': 'damage_2015'
}

df = df.rename(columns=cols)

# Basic checks
print('Rows:', len(df))
print(df[['masfem_coders','masfem_mturk','female_binary','deaths']].describe())

# Outcome transformation for skew

df['log_deaths'] = np.log1p(df['deaths'])

# Controls: use category, min_pressure, max_wind, log damage, year
# damage_2015 appears normalized; use log1p to reduce skew

df['log_damage_2015'] = np.log1p(df['damage_2015'])

# Drop rows with missing values in model vars
model_vars = ['log_deaths','masfem_coders','masfem_mturk','female_binary','category','min_pressure','max_wind','log_damage_2015','year']

print('Missing values per column:')
print(df[model_vars].isna().sum())

# Model 1: log_deaths ~ masfem_coders + controls
X1 = df[['masfem_coders','category','min_pressure','max_wind','log_damage_2015','year']].copy()
X1 = sm.add_constant(X1)
model1 = sm.OLS(df['log_deaths'], X1).fit()
print('\nModel 1 (masfem_coders + controls)')
print(model1.summary())

# Model 2: log_deaths ~ masfem_mturk + controls
X2 = df[['masfem_mturk','category','min_pressure','max_wind','log_damage_2015','year']].copy()
X2 = sm.add_constant(X2)
model2 = sm.OLS(df['log_deaths'], X2).fit()
print('\nModel 2 (masfem_mturk + controls)')
print(model2.summary())

# Model 3: log_deaths ~ female_binary + controls
X3 = df[['female_binary','category','min_pressure','max_wind','log_damage_2015','year']].copy()
X3 = sm.add_constant(X3)
model3 = sm.OLS(df['log_deaths'], X3).fit()
print('\nModel 3 (female_binary + controls)')
print(model3.summary())

# Bivariate correlations
print('\nCorrelations with log_deaths:')
for col in ['masfem_coders','masfem_mturk','female_binary']:
    corr = df['log_deaths'].corr(df[col])
    print(col, corr)

