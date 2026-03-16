import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('panda_nuts.csv')
df['log_seconds'] = np.log(df['seconds'])

# GEE with Negative Binomial to handle overdispersion and clustering by chimp
gee_nb = smf.gee('nuts_opened ~ age + C(sex) + C(help)', groups='chimpanzee', data=df,
                 family=sm.families.NegativeBinomial(), offset=df['log_seconds']).fit()
print('GEE Negative Binomial')
print(gee_nb.summary())

# GEE with Poisson robust (sandwich) for comparison
gee_pois = smf.gee('nuts_opened ~ age + C(sex) + C(help)', groups='chimpanzee', data=df,
                   family=sm.families.Poisson(), offset=df['log_seconds']).fit()
print('\nGEE Poisson')
print(gee_pois.summary())

