import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

path='panda_nuts.csv'
df=pd.read_csv(path)
df=df[df['seconds']>0].copy()
df['log_seconds']=np.log(df['seconds'])
model=smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df, family=sm.families.Poisson(), offset=df['log_seconds']).fit()
print('deviance', model.deviance, 'df_resid', model.df_resid, 'ratio', model.deviance/model.df_resid)
print('pearson', model.pearson_chi2, 'ratio', model.pearson_chi2/model.df_resid)

# Negative binomial as robustness if overdispersion
nb = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=df, family=sm.families.NegativeBinomial(alpha=1.0), offset=df['log_seconds']).fit()
print(nb.summary2().tables[1])
