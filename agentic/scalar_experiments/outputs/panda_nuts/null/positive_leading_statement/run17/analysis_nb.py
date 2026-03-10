import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('panda_nuts.csv')
for col in ['sex','help','hammer']:
    df[col] = df[col].astype('category')

df['log_seconds'] = np.log(df['seconds'])

# Negative Binomial (discrete) with offset
nb = smf.negativebinomial('nuts_opened ~ age + C(sex) + C(help)', data=df, offset=df['log_seconds']).fit(disp=False)
print(nb.summary())

# robust SEs
nb_robust = nb.get_robustcov_results(cov_type='HC3')
print('\nNegative Binomial (robust HC3)')
print(nb_robust.summary())

# IRR
params = nb.params
conf = nb.conf_int()
irr = np.exp(params)
conf_irr = np.exp(conf)
print('\nIRR and 95% CI')
for name in params.index:
    print(name, irr[name], conf_irr.loc[name, 0], conf_irr.loc[name, 1])

# effect sizes for efficiency OLS (robust)


