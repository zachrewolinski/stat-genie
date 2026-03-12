import pandas as pd
import numpy as np

path = 'amtl.csv'
df = pd.read_csv(path)

num_cols = ['genus','age','pop','num_amtl','stdev_age']

print('Group means by tooth_class (genus categorical):')
for col in num_cols:
    means = df.groupby('tooth_class')[col].mean().sort_values()
    print('\n', col)
    print(means)

print('\nGroup means by sockets (tooth class):')
for col in num_cols:
    means = df.groupby('sockets')[col].mean().sort_values()
    print('\n', col)
    print(means)

# check within-specimen variance for numeric columns
spec = 'prob_male'
for col in num_cols:
    within_var = df.groupby(spec)[col].var().dropna().mean()
    print(f'\n{col} mean within-specimen variance: {within_var:.4f}')

# check if age (integer) corresponds to sockets counts; distribution by sockets
print('\nAge (possible sockets) by sockets category:')
print(df.groupby('sockets')['age'].describe())

