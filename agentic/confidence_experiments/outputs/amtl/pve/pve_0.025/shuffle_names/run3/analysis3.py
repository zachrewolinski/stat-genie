import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

for c in ['genus','age','pop','num_amtl','stdev_age']:
    print('\nMean', c, 'by sockets(tooth_class):')
    print(df.groupby('sockets')[c].mean().sort_values())

# check if num_amtl <= age for most rows (maybe missing<=sockets)
print('\nFraction num_amtl <= age:', (df['num_amtl'] <= df['age']).mean())
print('Fraction genus <= age:', (df['genus'] <= df['age']).mean())

# show max/min by sockets for age and num_amtl
for c in ['age','num_amtl','genus']:
    print('\n', c, 'min/max by sockets')
    print(df.groupby('sockets')[c].agg(['min','max','mean']))
