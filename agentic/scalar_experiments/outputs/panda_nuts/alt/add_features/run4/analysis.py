import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('panda_nuts.csv')
print('rows', len(df))
print(df.head())

# compute efficiency as nuts opened per second
# avoid division by zero

df = df.copy()
df['efficiency'] = df['nuts_opened'] / df['seconds']

# basic summary
print(df[['nuts_opened','seconds','efficiency']].describe())
print(df['efficiency'].describe())
print('chimp unique', df['chimpanzee'].nunique())
print('rows per chimpanzee', df.groupby('chimpanzee').size().describe())

# check missing values
print('missing', df[['age','sex','help','nuts_opened','seconds']].isna().sum())

# OLS regression with categorical sex/help
# Ensure categories

df['sex'] = df['sex'].astype('category')
df['help'] = df['help'].astype('category')

# Use log efficiency to reduce skew, add small constant to handle zeros
# If any efficiency zeros, add epsilon

eps = 1e-6
if (df['efficiency']<=0).any():
    eps = 1e-3

df['log_eff'] = np.log(df['efficiency'] + eps)

ols = smf.ols('log_eff ~ age + sex + help', data=df).fit(cov_type='HC3')
print(ols.summary())

# Mixed effects random intercept for chimpanzee
try:
    mixed = smf.mixedlm('log_eff ~ age + sex + help', data=df, groups=df['chimpanzee']).fit()
    print(mixed.summary())
except Exception as e:
    print('MixedLM failed', e)

# Also GLM for counts with offset log(seconds) to model rate using Poisson
# Use nuts_opened as count. Add offset log(seconds)
import statsmodels.api as sm

# remove rows with zero seconds or missing
mask = (df['seconds']>0) & df['nuts_opened'].notna()

glm = smf.glm('nuts_opened ~ age + sex + help', data=df[mask],
              family=sm.families.Poisson(), offset=np.log(df.loc[mask, 'seconds'])).fit(cov_type='HC3')
print(glm.summary())

# Overdispersion check: ratio of deviance/df_resid
print('Poisson overdispersion', glm.deviance / glm.df_resid)

# Negative binomial if overdispersion
try:
    nb = smf.glm('nuts_opened ~ age + sex + help', data=df[mask],
                 family=sm.families.NegativeBinomial(alpha=1.0), offset=np.log(df.loc[mask, 'seconds'])).fit(cov_type='HC3')
    print(nb.summary())
except Exception as e:
    print('NB failed', e)

# effect sizes: exponentiate coefficients from Poisson to interpret as rate ratios
print('Poisson rate ratios', np.exp(glm.params))
print('Poisson pvalues', glm.pvalues)

# GEE with clustering by chimpanzee to account for repeated measures
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable

gee_poisson = GEE.from_formula(
    'nuts_opened ~ age + sex + help',
    groups='chimpanzee',
    data=df[mask],
    family=sm.families.Poisson(),
    cov_struct=Exchangeable(),
    offset=np.log(df.loc[mask, 'seconds'])
).fit()
print(gee_poisson.summary())

try:
    gee_nb = GEE.from_formula(
        'nuts_opened ~ age + sex + help',
        groups='chimpanzee',
        data=df[mask],
        family=sm.families.NegativeBinomial(),
        cov_struct=Exchangeable(),
        offset=np.log(df.loc[mask, 'seconds'])
    ).fit()
    print(gee_nb.summary())
except Exception as e:
    print('GEE NB failed', e)
