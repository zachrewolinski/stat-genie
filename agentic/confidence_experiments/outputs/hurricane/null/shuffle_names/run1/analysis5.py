import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

path = 'hurricane.csv'
df = pd.read_csv(path)

# Map variables

df['deaths'] = df['name']
df['log_deaths'] = np.log1p(df['deaths'])

df['fem_index'] = df['category']
df['fem_mturk'] = df['ind']
df['female_binary'] = df['masfem_mturk']

df['storm_year'] = df['wind']
df['wind_speed'] = df['year']
df['min_pressure'] = df['ndam15']
df['ss_category'] = df['gender_mf']

# Interaction models
ols_int = smf.ols('log_deaths ~ fem_index * wind_speed + min_pressure + ss_category + storm_year', data=df).fit()
print(ols_int.summary())

nb_int = smf.glm('deaths ~ fem_index * wind_speed + min_pressure + ss_category + storm_year', data=df, family=sm.families.NegativeBinomial()).fit()
print(nb_int.summary())

