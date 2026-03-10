import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import NegativeBinomial

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Create log deaths

df['log_deaths'] = np.log1p(df['alldeaths'])

# Correlation
corr_masfem_deaths = df['masfem'].corr(df['alldeaths'])

# OLS models
m1 = smf.ols('log_deaths ~ masfem', data=df).fit()
m2 = smf.ols('log_deaths ~ masfem + category + wind + min', data=df).fit()
m3 = smf.ols('log_deaths ~ gender_mf + category + wind + min', data=df).fit()
m4 = smf.ols('log_deaths ~ masfem + category + wind + min + year', data=df).fit()

# Poisson GLM with robust SE
poisson = smf.glm('alldeaths ~ masfem + category + wind + min', data=df, family=sm.families.Poisson()).fit(cov_type='HC0')

# Negative Binomial (discrete) with estimated alpha
X = sm.add_constant(df[['masfem','category','wind','min']])
y = df['alldeaths']
nb = NegativeBinomial(y, X).fit(disp=False)

# Sensitivity: exclude top 5% deaths
threshold = df['alldeaths'].quantile(0.95)
df_trim = df[df['alldeaths'] <= threshold].copy()
df_trim['log_deaths'] = np.log1p(df_trim['alldeaths'])

m2_trim = smf.ols('log_deaths ~ masfem + category + wind + min', data=df_trim).fit()
poisson_trim = smf.glm('alldeaths ~ masfem + category + wind + min', data=df_trim, family=sm.families.Poisson()).fit(cov_type='HC0')
X_trim = sm.add_constant(df_trim[['masfem','category','wind','min']])
y_trim = df_trim['alldeaths']
nb_trim = NegativeBinomial(y_trim, X_trim).fit(disp=False)

# Print summary stats
print('N=', len(df))
print('corr(masfem, alldeaths)=', corr_masfem_deaths)
print('m1 coef masfem', m1.params['masfem'], 'p', m1.pvalues['masfem'])
print('m2 coef masfem', m2.params['masfem'], 'p', m2.pvalues['masfem'])
print('m3 coef gender_mf', m3.params['gender_mf'], 'p', m3.pvalues['gender_mf'])
print('m4 coef masfem', m4.params['masfem'], 'p', m4.pvalues['masfem'])
print('poisson coef masfem', poisson.params['masfem'], 'p', poisson.pvalues['masfem'])
print('nb coef masfem', nb.params['masfem'], 'p', nb.pvalues['masfem'])
print('nb alpha', nb.params['alpha'])
print('trim N=', len(df_trim), 'threshold', threshold)
print('m2_trim coef masfem', m2_trim.params['masfem'], 'p', m2_trim.pvalues['masfem'])
print('poisson_trim coef masfem', poisson_trim.params['masfem'], 'p', poisson_trim.pvalues['masfem'])
print('nb_trim coef masfem', nb_trim.params['masfem'], 'p', nb_trim.pvalues['masfem'])

