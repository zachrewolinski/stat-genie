import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

pd.set_option('display.width', 120)

# Load data
df = pd.read_csv('panda_nuts.csv')

# Keep relevant columns
cols = ['age', 'sex', 'help', 'nuts_opened', 'seconds']
df = df[cols].copy()

# Clean
# Ensure valid seconds > 0
clean = df.dropna().copy()
clean = clean[clean['seconds'] > 0]

# Create efficiency (nuts per second)
clean['rate'] = clean['nuts_opened'] / clean['seconds']

print('Rows after cleaning:', len(clean))
print('\nRate summary:')
print(clean['rate'].describe())

print('\nMean rate by sex:')
print(clean.groupby('sex')['rate'].mean())

print('\nMean rate by help:')
print(clean.groupby('help')['rate'].mean())

# OLS model on rate
ols = smf.ols('rate ~ age + C(sex) + C(help)', data=clean).fit()
print('\nOLS on rate:')
print(ols.summary())

# Poisson GLM with offset for seconds (rate model)
clean['log_seconds'] = np.log(clean['seconds'])
poisson = smf.glm('nuts_opened ~ age + C(sex) + C(help)',
                  data=clean,
                  family=sm.families.Poisson(),
                  offset=clean['log_seconds']).fit()
print('\nPoisson GLM with offset:')
print(poisson.summary())

# Overdispersion check for Poisson
pearson_chi2 = ((poisson.resid_pearson**2).sum())
print('\nPoisson dispersion (Pearson chi2/df):', pearson_chi2 / poisson.df_resid)

# Negative binomial GLM if overdispersed
nb = smf.glm('nuts_opened ~ age + C(sex) + C(help)',
             data=clean,
             family=sm.families.NegativeBinomial(),
             offset=clean['log_seconds']).fit()
print('\nNegative Binomial GLM with offset:')
print(nb.summary())

# Rate ratios from GLM (exp coefficients)
poisson_rr = np.exp(poisson.params)
nb_rr = np.exp(nb.params)
print('\nPoisson rate ratios:')
print(poisson_rr)
print('\nNegative binomial rate ratios:')
print(nb_rr)
