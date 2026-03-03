import pandas as pd
import numpy as np

df=pd.read_csv('amtl.csv')

# sum genus by specimen
sum_genus = df.groupby('prob_male')['genus'].sum()
# sum age by specimen
sum_age = df.groupby('prob_male')['age'].sum()

spec=df.groupby('prob_male').first()[['num_amtl','pop','stdev_age','tooth_class','specimen']]

spec = spec.join(sum_genus.rename('sum_genus')).join(sum_age.rename('sum_age'))

print(spec.head())

# correlations
print('\nCorrelations with num_amtl:')
for col in ['sum_genus','sum_age','pop']:
    print(col, spec['num_amtl'].corr(spec[col]))

print('\nCorrelations with pop:')
for col in ['sum_genus','sum_age','num_amtl']:
    print(col, spec['pop'].corr(spec[col]))

