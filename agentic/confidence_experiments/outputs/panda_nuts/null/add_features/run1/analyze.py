import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

path = 'panda_nuts.csv'
df = pd.read_csv(path)

# compute efficiency (nuts per second)
df['efficiency'] = df['nuts_opened'] / df['seconds']

print('rows', len(df))
print('efficiency summary')
print(df['efficiency'].describe())

# OLS with categorical sex/help
ols = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit()
print('\nOLS results')
print(ols.summary())

# log-transformed efficiency to reduce skew (add small constant)
df['log_eff'] = np.log(df['efficiency'] + 1e-6)
ols_log = smf.ols('log_eff ~ age + C(sex) + C(help)', data=df).fit()
print('\nOLS log-eff results')
print(ols_log.summary())

# Poisson GLM for nuts_opened with log(seconds) offset
# This models rate while keeping count nature
poisson = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['seconds'])
).fit()
print('\nPoisson GLM results')
print(poisson.summary())

# Check overdispersion and, if needed, fit Negative Binomial
# (variance >> mean indicates overdispersion)
mean_nuts = df['nuts_opened'].mean()
var_nuts = df['nuts_opened'].var()
print(f"\nNuts opened mean={mean_nuts:.3f}, var={var_nuts:.3f}")

nb = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=np.log(df['seconds'])
).fit()
print('\nNegative Binomial GLM results')
print(nb.summary())

# group means for intuition
print('\nmean efficiency by sex')
print(df.groupby('sex')['efficiency'].mean())
print('\nmean efficiency by help')
print(df.groupby('help')['efficiency'].mean())
