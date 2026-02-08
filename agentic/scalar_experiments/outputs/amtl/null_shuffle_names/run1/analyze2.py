import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

print('unique stdev_age', sorted(df['stdev_age'].unique()))

# correlations
num_cols = df.select_dtypes(include='number')
print(num_cols.corr())

# check if num_amtl close to pop? 
print('num_amtl min/max', df['num_amtl'].min(), df['num_amtl'].max())
print('pop min/max', df['pop'].min(), df['pop'].max())

# check if age or genus integer counts
print('age unique sample', sorted(df['age'].unique())[:15])
print('genus unique sample', sorted(df['genus'].unique())[:15])

# check if age or genus maybe number of sockets by tooth class (anterior 6, posterior 8, premolar 4?)
# typical adult human: anterior 12, premolar 8, posterior 12? hmm.
# We'll check max per tooth class by sockets category.
for col in ['genus','age']:
    print('\n', col)
    print(df.groupby('sockets')[col].max())
    print(df.groupby('sockets')[col].min())

