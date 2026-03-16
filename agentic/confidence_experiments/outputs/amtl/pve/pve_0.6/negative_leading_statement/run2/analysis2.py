import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

# compute expit and multiply by sockets
p = 1/(1+np.exp(-df['num_amtl']))
expected = p * df['sockets']
# check how close to integer
rounded = expected.round()
err = np.abs(expected - rounded)
print('mean abs error', err.mean())
print('pct within 0.1', (err < 0.1).mean())
print('pct within 0.01', (err < 0.01).mean())
print('example rows')
print(expected.head())

# also check if num_amtl maybe arcsin sqrt proportion? -> transform back
p2 = np.sin(df['num_amtl'])**2
expected2 = p2 * df['sockets']
err2 = np.abs(expected2 - expected2.round())
print('arcsin mean abs err', err2.mean())
print('arcsin pct within 0.1', (err2 < 0.1).mean())

# check if num_amtl maybe standardized count; reverse: look for possible integer counts by solving linear transform
# assume num_amtl = (count - mu)/sigma. Try to infer mu and sigma from min/max? not reliable.

