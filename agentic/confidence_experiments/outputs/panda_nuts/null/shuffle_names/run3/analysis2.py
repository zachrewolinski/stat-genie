import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

pd.set_option('display.width', 120)

# Load data
raw = pd.read_csv('panda_nuts.csv')

# Map columns to semantic variables (based on value patterns)
# age: numeric 1-22
# sex: stored in 'nuts_opened' (f/m)
# help indicator: stored in 'seconds' (y/N)
# nuts opened count: stored in 'help'
# session duration (seconds): stored in 'chimpanzee'

df = raw.copy()
df['sex_true'] = df['nuts_opened'].map({'f': 'f', 'm': 'm'})
df['help_true'] = df['seconds'].map({'y': 1, 'N': 0})
df['nuts_opened_count'] = df['help']
df['duration_seconds'] = df['chimpanzee']

# Efficiency: nuts opened per second
# Avoid division by zero

df = df[df['duration_seconds'] > 0].copy()
df['efficiency'] = df['nuts_opened_count'] / df['duration_seconds']

# Basic summaries
summary = df[['age', 'sex_true', 'help_true', 'nuts_opened_count', 'duration_seconds', 'efficiency']].describe(include='all')
print(summary)

# Group means
print('\nMean efficiency by sex')
print(df.groupby('sex_true')['efficiency'].mean())
print('\nMean efficiency by help')
print(df.groupby('help_true')['efficiency'].mean())

# OLS regression with robust SEs
ols_model = smf.ols('efficiency ~ age + C(sex_true) + help_true', data=df).fit(cov_type='HC3')
print('\nOLS model (HC3)')
print(ols_model.summary())

# Spearman correlation for age vs efficiency
rho, pval = stats.spearmanr(df['age'], df['efficiency'])
print('\nSpearman age-efficiency:', rho, pval)

# Poisson GLM with offset for duration
# Model nuts_opened_count as rate * duration
poisson_model = smf.glm(
    'nuts_opened_count ~ age + C(sex_true) + help_true',
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['duration_seconds'])
).fit(cov_type='HC3')
print('\nPoisson GLM with offset (HC3)')
print(poisson_model.summary())

# Overdispersion check
# ratio of Pearson chi2 to df
pearson_chi2 = sum(poisson_model.resid_pearson**2)
ratio = pearson_chi2 / poisson_model.df_resid
print('\nOverdispersion ratio:', ratio)

# Negative binomial model with offset (to handle overdispersion)
exog = pd.get_dummies(df[['age', 'sex_true', 'help_true']], columns=['sex_true'], drop_first=True)
exog = sm.add_constant(exog)
nb_model = sm.NegativeBinomial(
    df['nuts_opened_count'],
    exog,
    offset=np.log(df['duration_seconds'])
).fit(disp=False, cov_type='HC3')
print('\nNegative Binomial with offset (HC3)')
print(nb_model.summary())
