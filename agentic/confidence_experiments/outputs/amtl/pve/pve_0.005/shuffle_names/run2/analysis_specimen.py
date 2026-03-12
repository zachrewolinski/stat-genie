import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('amtl.csv')

# construct specimen-level dataset
# total sockets from age column per tooth class
sockets_wide = df.pivot_table(index='prob_male', columns='sockets', values='age', aggfunc='first')

spec = df.groupby('prob_male').first()[['tooth_class','pop','stdev_age','num_amtl']].join(sockets_wide)

spec = spec.rename(columns={'tooth_class':'genus_cat','pop':'age_at_death','stdev_age':'sex_prob','num_amtl':'num_amtl_total',
                            'Anterior':'sockets_anterior','Posterior':'sockets_posterior','Premolar':'sockets_premolar'})

# total sockets
spec['total_sockets'] = spec[['sockets_anterior','sockets_posterior','sockets_premolar']].sum(axis=1)

# amtl rate
spec['amtl_rate'] = spec['num_amtl_total'] / spec['total_sockets']

spec['is_human'] = (spec['genus_cat']=='Homo sapiens').astype(int)

# OLS on rate with tooth class sockets as covariates
model = smf.ols('amtl_rate ~ is_human + age_at_death + sex_prob + sockets_anterior + sockets_posterior + sockets_premolar', data=spec).fit(cov_type='HC3')
print(model.summary())

