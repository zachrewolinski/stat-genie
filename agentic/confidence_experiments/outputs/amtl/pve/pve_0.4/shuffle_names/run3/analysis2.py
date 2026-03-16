import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

for col in ['genus','age','pop','num_amtl','stdev_age']:
    vals = df[col].values
    if pd.api.types.is_numeric_dtype(df[col]):
        diff = np.abs(vals - np.round(vals))
        print(col, 'mean abs diff to integer', diff.mean(), 'min', vals.min(), 'max', vals.max())

print('num_amtl summary', df['num_amtl'].describe())
print('genus summary', df['genus'].describe())
print('age summary', df['age'].describe())
print('pop summary', df['pop'].describe())

# check relationship with sockets: mean by sockets
print('\nmean genus by sockets')
print(df.groupby('sockets')['genus'].mean())
print('\nmean num_amtl by sockets')
print(df.groupby('sockets')['num_amtl'].mean())

# check if num_amtl could be age at death? correlate with pop
print('corr num_amtl pop', df['num_amtl'].corr(df['pop']))
print('corr genus pop', df['genus'].corr(df['pop']))

# examine stdev_age distribution by tooth_class
print(df.groupby('tooth_class')['stdev_age'].mean())

# unique values in age
print('age unique', sorted(df['age'].unique())[:20])
