import pandas as pd

df = pd.read_csv('amtl.csv')

sockets_wide = df.pivot_table(index='prob_male', columns='sockets', values='age', aggfunc='first')

spec = df.groupby('prob_male').first()[['tooth_class','pop','stdev_age','num_amtl']].join(sockets_wide)

spec = spec.rename(columns={'tooth_class':'genus_cat','pop':'age_at_death','stdev_age':'sex_prob','num_amtl':'num_amtl_total',
                            'Anterior':'sockets_anterior','Posterior':'sockets_posterior','Premolar':'sockets_premolar'})

spec['total_sockets'] = spec[['sockets_anterior','sockets_posterior','sockets_premolar']].sum(axis=1)

spec['amtl_rate'] = spec['num_amtl_total'] / spec['total_sockets']

print(spec.groupby('genus_cat')['amtl_rate'].describe())

