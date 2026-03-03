import pandas as pd
import numpy as np


df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)

# check if num_amtl near integers
num = df['num_amtl']
int_diff = np.abs(num - np.round(num))
print('mean abs diff to integer:', int_diff.mean())
print('min/max num_amtl:', num.min(), num.max())

# check if num_amtl within [0, sockets]
print('min num_amtl - sockets max', (num - df['sockets']).max())
print('min num_amtl - sockets min', (num - df['sockets']).min())
print('num_amtl proportion range', (num/df['sockets']).min(), (num/df['sockets']).max())

# check counts of genus
print(df['genus'].value_counts())
print(df['tooth_class'].value_counts())

