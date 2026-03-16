import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

path='panda_nuts.csv'
df=pd.read_csv(path)
df=df[df['seconds']>0].copy()
df['log_seconds']=np.log(df['seconds'])
model=smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df, family=sm.families.Poisson(), offset=df['log_seconds']).fit(cov_type='HC0')

params = model.params
conf = model.conf_int()

# rate ratios for age, sex (male vs female), help (y vs N)
for term in ['age', 'C(sex)[T.m]', 'C(help)[T.y]']:
    rr = float(np.exp(params[term]))
    lo = float(np.exp(conf.loc[term, 0]))
    hi = float(np.exp(conf.loc[term, 1]))
    p = float(model.pvalues[term])
    print(term, rr, lo, hi, p)
