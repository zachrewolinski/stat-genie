import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

path='panda_nuts.csv'
df=pd.read_csv(path)
df=df[df['seconds']>0].copy()
df['log_seconds']=np.log(df['seconds'])
model=smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df, family=sm.families.Poisson(), offset=df['log_seconds']).fit(cov_type='HC0')
print(model.summary2().tables[1])

