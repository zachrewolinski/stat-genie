import pandas as pd
import numpy as np

# Load data
amtl = pd.read_csv('amtl.csv')
print(amtl.head())
print(amtl.dtypes)
print(amtl.describe(include='all'))
print('rows', len(amtl))
print(amtl['genus'].value_counts())
print(amtl['tooth_class'].value_counts())

# Check num_amtl ranges vs sockets
print('num_amtl min/max', amtl['num_amtl'].min(), amtl['num_amtl'].max())
print('sockets min/max', amtl['sockets'].min(), amtl['sockets'].max())
print('any negative num_amtl', (amtl['num_amtl']<0).sum())

# check if num_amtl is integer? or transformed
print('num_amtl unique sample', amtl['num_amtl'].head(10).tolist())
print('num_amtl fractional', (amtl['num_amtl']%1!=0).sum())

# maybe standardized? check mean and std by genus
print(amtl.groupby('genus')['num_amtl'].agg(['mean','std','min','max']))

# Evaluate if num_amtl corresponds to (missing / sockets) maybe standardized
amtl['amtl_rate']=amtl['num_amtl']/amtl['sockets']
print(amtl['amtl_rate'].describe())

# check per genus amtl_rate
print(amtl.groupby('genus')['amtl_rate'].agg(['mean','std','min','max']))

# Are there negative num_amtl? if so maybe it's residualized? check rounding? Try reconstruct using stored? if num_amtl negative maybe mean centered? Use columns? maybe not.

