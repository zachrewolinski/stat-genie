import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import NegativeBinomial

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Basic cleaning
# Ensure categorical variables
for col in ['sex', 'help', 'hammer']:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Efficiency: nuts per second
# Avoid division by zero (seconds min >0 per metadata)
df['efficiency'] = df['nuts_opened'] / df['seconds']

# OLS model on efficiency
ols_model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')

# Log-efficiency (add 1 nut to allow zeros)
df['log_efficiency'] = np.log((df['nuts_opened'] + 1) / df['seconds'])
log_ols_model = smf.ols('log_efficiency ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')

# Poisson GLM for counts with offset log(seconds)
# Using Poisson; check for overdispersion via deviance/df
poisson_model = smf.glm('nuts_opened ~ age + C(sex) + C(help)',
                        data=df,
                        family=sm.families.Poisson(),
                        offset=np.log(df['seconds'])).fit()
poisson_robust = smf.glm('nuts_opened ~ age + C(sex) + C(help)',
                         data=df,
                         family=sm.families.Poisson(),
                         offset=np.log(df['seconds'])).fit(cov_type='HC3')

# Overdispersion metric
od = poisson_model.deviance / poisson_model.df_resid

# Negative binomial with offset (NB2)
exog = sm.add_constant(pd.get_dummies(df[['age', 'sex', 'help']], drop_first=True))
nb_model = NegativeBinomial(df['nuts_opened'], exog, offset=np.log(df['seconds'])).fit(disp=False)

# Summaries
print('N:', len(df))
print('Efficiency summary:')
print(df['efficiency'].describe())
print('\nOLS (HC3) coefficients:')
print(ols_model.summary())
print('\nLog-efficiency OLS (HC3) coefficients:')
print(log_ols_model.summary())
print('\nPoisson coefficients:')
print(poisson_model.summary())
print('\nPoisson coefficients (HC3 robust SE):')
print(poisson_robust.summary())
print('\nOverdispersion (deviance/df_resid):', od)
print('\nNegative binomial coefficients:')
print(nb_model.summary())

# Simple group means for sex and help
print('\nGroup means efficiency by sex:')
print(df.groupby('sex')['efficiency'].mean())
print('\nGroup means efficiency by help:')
print(df.groupby('help')['efficiency'].mean())

# Correlation with age
print('\nCorrelation age vs efficiency:')
print(df[['age', 'efficiency']].corr().iloc[0,1])
