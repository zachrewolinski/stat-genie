import pandas as pd
import numpy as np

amtl = pd.read_csv('amtl.csv')

# compute logistic transform of num_amtl
p = 1/(1+np.exp(-amtl['num_amtl']))
expected = p * amtl['sockets']
# check how close to integer
frac = np.abs(expected - np.round(expected))
print('mean abs fractional', frac.mean())
print('median abs fractional', frac.median())
print('<=0.05 fraction', (frac<=0.05).mean())
print('<=0.1 fraction', (frac<=0.1).mean())
print('min max expected', expected.min(), expected.max())

# check if simple proportion or log proportion? maybe num_amtl / sockets? but num_amtl negative.

# maybe num_amtl is z-scored counts
mean = amtl['num_amtl'].mean()
std = amtl['num_amtl'].std(ddof=0)
print('num_amtl mean', mean, 'std', std)

# check if num_amtl * std + mean yields plausible integer? try find scaling to integer by linear transform with sockets?

# quick check if num_amtl correlates with sockets
print('corr num_amtl vs sockets', amtl['num_amtl'].corr(amtl['sockets']))

