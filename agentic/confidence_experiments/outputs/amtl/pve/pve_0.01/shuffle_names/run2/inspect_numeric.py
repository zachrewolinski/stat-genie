import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

for col in ['genus','age','pop','num_amtl','stdev_age']:
    vals = df[col]
    # check how many are near integer
    frac = np.abs(vals - np.round(vals))
    near_int = (frac < 1e-6).mean()
    print(col, 'near_int%', near_int, 'min', vals.min(), 'max', vals.max())

# check unique values counts for age and stdev_age
print('age uniques', sorted(df['age'].unique())[:20])
print('stdev_age uniques', sorted(df['stdev_age'].unique()))

# check num_amtl unique count
print('num_amtl unique count', df['num_amtl'].nunique())
print('genus unique count', df['genus'].nunique())
print('pop unique count', df['pop'].nunique())
