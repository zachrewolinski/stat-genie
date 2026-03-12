import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_hc3

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

print('Deaths mean', df['deaths'].mean(), 'variance', df['deaths'].var())

# OLS with robust SE
ols = smf.ols('log_deaths ~ fem_index + wind_speed + min_pressure + ss_category + storm_year', data=df).fit()
robust_se = np.sqrt(np.diag(cov_hc3(ols)))
print('OLS fem_index coef', ols.params['fem_index'], 'HC3 SE', robust_se[ols.params.index.get_loc('fem_index')], 'p', ols.pvalues['fem_index'])

ols2 = smf.ols('log_deaths ~ fem_mturk + wind_speed + min_pressure + ss_category + storm_year', data=df).fit()
robust_se2 = np.sqrt(np.diag(cov_hc3(ols2)))
print('OLS fem_mturk coef', ols2.params['fem_mturk'], 'HC3 SE', robust_se2[ols2.params.index.get_loc('fem_mturk')], 'p', ols2.pvalues['fem_mturk'])

ols3 = smf.ols('log_deaths ~ female_binary + wind_speed + min_pressure + ss_category + storm_year', data=df).fit()
robust_se3 = np.sqrt(np.diag(cov_hc3(ols3)))
print('OLS female_binary coef', ols3.params['female_binary'], 'HC3 SE', robust_se3[ols3.params.index.get_loc('female_binary')], 'p', ols3.pvalues['female_binary'])

# Negative binomial (GLM)
nb1 = smf.glm('deaths ~ fem_index + wind_speed + min_pressure + ss_category + storm_year', data=df, family=sm.families.NegativeBinomial()).fit()
print('NB fem_index', nb1.params['fem_index'], 'p', nb1.pvalues['fem_index'])

nb2 = smf.glm('deaths ~ fem_mturk + wind_speed + min_pressure + ss_category + storm_year', data=df, family=sm.families.NegativeBinomial()).fit()
print('NB fem_mturk', nb2.params['fem_mturk'], 'p', nb2.pvalues['fem_mturk'])

nb3 = smf.glm('deaths ~ female_binary + wind_speed + min_pressure + ss_category + storm_year', data=df, family=sm.families.NegativeBinomial()).fit()
print('NB female_binary', nb3.params['female_binary'], 'p', nb3.pvalues['female_binary'])

