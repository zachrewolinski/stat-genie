import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('panda_nuts.csv')

# basic checks
print('rows', len(df))
print(df.head())
print(df.dtypes)

# clean categorical
for col in ['sex','help','hammer']:
    df[col] = df[col].astype('category')

# efficiency: nuts opened per second
# avoid divide by zero (none expected)
df['efficiency'] = df['nuts_opened'] / df['seconds']

print('\nEfficiency summary:')
print(df['efficiency'].describe())

# OLS on efficiency
ols = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')
print('\nOLS efficiency ~ age + sex + help (HC3):')
print(ols.summary())

# Poisson count model with offset for seconds
# add 1e-6 to seconds just in case
df['log_seconds'] = np.log(df['seconds'])
poisson = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
                  family=sm.families.Poisson(), offset=df['log_seconds']).fit()
print('\nPoisson count with offset log(seconds):')
print(poisson.summary())

# check overdispersion for Poisson
mu = poisson.fittedvalues
resid = df['nuts_opened'] - mu
phi = (resid**2).sum() / poisson.df_resid
print('\nOverdispersion phi:', phi)

# Negative binomial if overdispersed
nb = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
             family=sm.families.NegativeBinomial(alpha=1.0), offset=df['log_seconds']).fit()
print('\nNegBin (alpha=1.0) count with offset:')
print(nb.summary())

# Group means for sex/help
print('\nGroup means efficiency by sex:')
print(df.groupby('sex')['efficiency'].mean())
print('\nGroup means efficiency by help:')
print(df.groupby('help')['efficiency'].mean())

# simple correlations
print('\nSpearman age vs efficiency:', df['age'].corr(df['efficiency'], method='spearman'))
print('Pearson age vs efficiency:', df['age'].corr(df['efficiency'], method='pearson'))
