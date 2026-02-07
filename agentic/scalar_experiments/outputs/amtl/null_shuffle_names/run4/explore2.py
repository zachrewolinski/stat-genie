import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

for col in ['genus','age','pop','num_amtl','stdev_age']:
    vals = df[col].values
    frac = np.abs(vals - np.round(vals))
    print(col, 'unique frac:', np.unique(frac)[:10], 'frac!=0 count', (frac>1e-6).sum())

print('genus min/max', df['genus'].min(), df['genus'].max())
print('age min/max', df['age'].min(), df['age'].max())
print('pop min/max', df['pop'].min(), df['pop'].max())
print('num_amtl min/max', df['num_amtl'].min(), df['num_amtl'].max())
print('stdev_age unique', sorted(df['stdev_age'].unique()))

print('tooth_class unique', df['tooth_class'].unique())
print('sockets unique', df['sockets'].unique())
