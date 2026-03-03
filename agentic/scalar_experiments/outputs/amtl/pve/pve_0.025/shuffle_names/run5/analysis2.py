import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print(df[['genus','age','pop','num_amtl','stdev_age']].describe())
for col in ['genus','age','pop','num_amtl','stdev_age']:
    s = df[col]
    print(col, 'min', s.min(), 'max', s.max(), 'unique sample', s.unique()[:10])

# check whether any columns are close to integers
for col in ['genus','pop','num_amtl','stdev_age']:
    s = df[col]
    frac = np.mean(np.isclose(s, np.round(s)))
    print(col, 'fraction near integer', frac)

# check if num_amtl <= age (if those are missing vs sockets)
print('num_amtl <= age fraction', np.mean(df['num_amtl'] <= df['age']))
print('genus <= age fraction', np.mean(df['genus'] <= df['age']))

print('rows per specimen', df.groupby('prob_male').size().value_counts().head())
