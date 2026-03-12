import pandas as pd
import numpy as np

_df = pd.read_csv('amtl.csv')

# rename for clarity
_df = _df.rename(columns={
    'sockets':'tooth_class',
    'prob_male':'specimen_id',
    'genus':'amtl_value',
    'age':'observable_sockets',
    'pop':'age_est',
    'num_amtl':'age_uncert',
    'stdev_age':'prob_male',
    'tooth_class':'genus',
    'specimen':'region'
})

print(_df.head())

# correlations
for col in ['observable_sockets','age_est','age_uncert','prob_male']:
    corr = _df['amtl_value'].corr(_df[col])
    print('corr amtl_value with', col, corr)

print('amtl_value by genus mean/std:')
print(_df.groupby('genus')['amtl_value'].agg(['mean','std','count']))

print('observable_sockets by genus mean:')
print(_df.groupby('genus')['observable_sockets'].mean())

# check if amtl_value close to integer count? unique counts per specimen+class? See decimals
print('amtl_value decimals unique:', len(_df['amtl_value'].unique()))
print('amtl_value near integer proportion:', np.mean(np.isclose(_df['amtl_value'], np.round(_df['amtl_value']))))

# check distribution of amtl_value by tooth class
print('amtl_value by tooth_class mean:')
print(_df.groupby('tooth_class')['amtl_value'].mean())

