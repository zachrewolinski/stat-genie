import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('panda_nuts.csv')

# offset: log(seconds)
# Avoid zero seconds (none?)
if (df['seconds'] <= 0).any():
    raise ValueError('Nonpositive seconds found')

df['log_seconds'] = np.log(df['seconds'])

# Poisson GLM with offset
poisson_model = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
                        family=sm.families.Poisson(), offset=df['log_seconds']).fit()
print('Poisson GLM')
print(poisson_model.summary())

# Check overdispersion: Pearson chi2 / df
pearson_chi2 = ((poisson_model.resid_pearson)**2).sum()
dispersion = pearson_chi2 / poisson_model.df_resid
print('\nPoisson overdispersion ratio:', dispersion)

# Negative Binomial GLM
nb_model = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df,
                   family=sm.families.NegativeBinomial(alpha=1.0), offset=df['log_seconds']).fit()
print('\nNegative Binomial GLM (alpha fixed=1)')
print(nb_model.summary())

# Estimate alpha via NB2 using statsmodels discrete? Use GLM with NB and estimate alpha?
# We'll try discrete NegativeBinomial which estimates alpha but doesn't allow offset easily.
# Instead, use GLM NB with scale estimate from Pearson.

