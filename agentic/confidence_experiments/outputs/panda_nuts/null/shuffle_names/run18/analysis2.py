import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

raw = pd.read_csv('panda_nuts.csv')

df = pd.DataFrame({
    'age': raw['age'].astype(float),
    'sex': raw['nuts_opened'].astype(str),
    'help_received': raw['seconds'].astype(str),
    'nuts_opened': raw['help'].astype(float),
    'session_seconds': raw['chimpanzee'].astype(float),
})

df['help_received'] = df['help_received'].str.strip().str.lower().map({'y': 1, 'n': 0})
clean = df.dropna().copy()
clean['log_seconds'] = np.log(clean['session_seconds'])
clean['efficiency'] = clean['nuts_opened'] / clean['session_seconds']

# Poisson with robust SE
poisson = smf.glm('nuts_opened ~ age + C(sex) + help_received', data=clean,
                  family=sm.families.Poisson(), offset=clean['log_seconds']).fit()
poisson_robust = smf.glm('nuts_opened ~ age + C(sex) + help_received', data=clean,
                         family=sm.families.Poisson(), offset=clean['log_seconds']).fit(cov_type='HC0')

# Quasi-Poisson (scale by Pearson chi2/df)
pearson_chi2 = sum(poisson.resid_pearson**2)
ratio = pearson_chi2 / poisson.df_resid
poisson_scaled = smf.glm('nuts_opened ~ age + C(sex) + help_received', data=clean,
                         family=sm.families.Poisson(), offset=clean['log_seconds']).fit(scale=ratio)

# Discrete Negative Binomial (estimates alpha)
# Use statsmodels.discrete (offset allowed)
import statsmodels.discrete.discrete_model as smd

# Design matrices
import patsy

y, X = patsy.dmatrices('nuts_opened ~ age + C(sex) + help_received', data=clean, return_type='dataframe')
nb2 = smd.NegativeBinomial(y, X, loglike_method='nb2', offset=clean['log_seconds']).fit(disp=False)

print('Poisson robust (HC0)')
print(poisson_robust.summary())
print('\nPoisson scaled (quasi) scale:', ratio)
print(poisson_scaled.summary())
print('\nNegBin (discrete, alpha estimated)')
print(nb2.summary())

# Rate ratios for key effects (exp(coef)) with robust SE (Poisson)
print('\nRate ratios (Poisson robust)')
import numpy as np
params = poisson_robust.params
conf = poisson_robust.conf_int()
rr = np.exp(params)
rr_ci = np.exp(conf)
for k in params.index:
    print(k, 'RR', rr[k], 'CI', tuple(rr_ci.loc[k]))
