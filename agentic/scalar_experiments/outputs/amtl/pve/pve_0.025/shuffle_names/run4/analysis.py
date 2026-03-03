import pandas as pd
import numpy as np

path='amtl.csv'
df=pd.read_csv(path)
print(df.head())
print('unique sockets', df['sockets'].unique())
print('unique tooth_class', df['tooth_class'].unique())
print('unique specimen count', df['specimen'].nunique())
print('age min max', df['age'].min(), df['age'].max(), 'unique', df['age'].nunique())
print('pop min max', df['pop'].min(), df['pop'].max())
print('num_amtl min max', df['num_amtl'].min(), df['num_amtl'].max())
print('genus min max', df['genus'].min(), df['genus'].max())
print('stdev_age unique', sorted(df['stdev_age'].unique())[:10])

# Check if num_amtl or age are integers when rounded
for col in ['num_amtl','genus','pop']:
    vals = df[col]
    print(col, 'mean', vals.mean(), 'std', vals.std())
    # fraction close to integer
    frac_int = np.mean(np.isclose(vals, np.round(vals)))
    print(col, 'frac_int', frac_int)

# check relationship between age and pop
print('corr age-pop', df['age'].corr(df['pop']))
print('corr num_amtl-pop', df['num_amtl'].corr(df['pop']))
print('corr genus-pop', df['genus'].corr(df['pop']))
print('corr genus-num_amtl', df['genus'].corr(df['num_amtl']))

# check mean values by sockets (tooth class)
print('mean age by sockets')
print(df.groupby('sockets')['age'].mean())
print('mean num_amtl by sockets')
print(df.groupby('sockets')['num_amtl'].mean())
print('mean genus by sockets')
print(df.groupby('sockets')['genus'].mean())

# check mean by tooth_class (genus)
print('mean num_amtl by tooth_class')
print(df.groupby('tooth_class')['num_amtl'].mean())
print('mean genus by tooth_class')
print(df.groupby('tooth_class')['genus'].mean())

