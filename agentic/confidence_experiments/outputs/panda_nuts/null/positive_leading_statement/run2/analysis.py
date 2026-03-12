import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

path = 'panda_nuts.csv'
df = pd.read_csv(path)
print(df.head())
print(df.describe(include='all'))
print(df.dtypes)

# compute efficiency rate: nuts per second
# avoid division by zero

df['efficiency'] = df['nuts_opened'] / df['seconds']
print(df['efficiency'].describe())

# categorical coding

df['sex'] = df['sex'].astype('category')
df['help'] = df['help'].astype('category')

# OLS on log efficiency (add small constant)

df['log_eff'] = np.log(df['efficiency'] + 1e-6)
model = smf.ols('log_eff ~ age + C(sex) + C(help)', data=df).fit()
print(model.summary())

# GLM Poisson with offset log(seconds) for counts

df['log_seconds'] = np.log(df['seconds'])
model_pois = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df, family=sm.families.Poisson(), offset=df['log_seconds']).fit()
print(model_pois.summary())

# Negative binomial
model_nb = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df, family=sm.families.NegativeBinomial(), offset=df['log_seconds']).fit()
print(model_nb.summary())

# check overdispersion
mu = model_pois.fittedvalues
var = ((df['nuts_opened'] - mu)**2).mean()
print('mean fitted', mu.mean(), 'mean obs', df['nuts_opened'].mean(), 'mean squared residual', var)

# group means by help/sex
print(df.groupby('help')['efficiency'].mean())
print(df.groupby('sex')['efficiency'].mean())
print(df.groupby(['sex','help'])['efficiency'].mean())

