import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm


df = pd.read_csv('panda_nuts.csv')
df['log_seconds'] = np.log(df['seconds'])

gee_nb = smf.gee('nuts_opened ~ age + C(sex) + C(help)', groups='chimpanzee', data=df,
                 family=sm.families.NegativeBinomial(), offset=df['log_seconds']).fit()

# Extract coef, p, and 95% CI
params = gee_nb.params
conf = gee_nb.conf_int()

for term in params.index:
    coef = params[term]
    p = gee_nb.pvalues[term]
    lo, hi = conf.loc[term]
    print(term, coef, p, lo, hi, np.exp(coef), np.exp(lo), np.exp(hi))

