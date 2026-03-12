import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

df = pd.read_csv('panda_nuts.csv')
df['sex'] = df['sex'].astype('category')
df['help'] = df['help'].astype('category')

model = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df, family=sm.families.Poisson(), offset=np.log(df['seconds']))
res = model.fit(cov_type='cluster', cov_kwds={'groups': df['chimpanzee']})

params = res.params
conf = res.conf_int()

for name in ['age','C(sex)[T.m]','C(help)[T.y]']:
    irr = np.exp(params[name])
    lo = np.exp(conf.loc[name,0])
    hi = np.exp(conf.loc[name,1])
    print(name, irr, lo, hi, res.pvalues[name])
