import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

path = 'panda_nuts.csv'

df = pd.read_csv(path)
print(df.head())
print(df.describe(include='all'))

# efficiency = nuts per second
# guard division by zero

df['efficiency'] = df['nuts_opened'] / df['seconds']
print('efficiency summary', df['efficiency'].describe())

# OLS on efficiency
model_ols = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')
print(model_ols.summary())

# Poisson GLM on nuts_opened with offset log(seconds)
df['log_seconds'] = np.log(df['seconds'])
model_pois = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
                     family=sm.families.Poisson(), offset=df['log_seconds']).fit(cov_type='HC3')
print(model_pois.summary())

# Check overdispersion
mu = model_pois.fittedvalues
var = model_pois.resid_pearson.var()
print('Pearson residual variance', var)

# Negative binomial
model_nb = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
                   family=sm.families.NegativeBinomial(alpha=1.0), offset=df['log_seconds']).fit(cov_type='HC3')
print(model_nb.summary())

# Likelihood ratio / compare? We'll just compute p-values.

# Also try mixed model? Use cluster robust by chimpanzee for OLS and Poisson
model_ols_cluster = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(cov_type='cluster', cov_kwds={'groups': df['chimpanzee']})
print('OLS cluster')
print(model_ols_cluster.summary())

model_pois_cluster = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
                            family=sm.families.Poisson(), offset=df['log_seconds']).fit(cov_type='cluster', cov_kwds={'groups': df['chimpanzee']})
print('Poisson cluster')
print(model_pois_cluster.summary())

# Also test joint significance? Use Wald test for all predictors

# This is more manual for OLS
from statsmodels.stats.anova import anova_lm
print('ANOVA OLS')
print(anova_lm(smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit()))
