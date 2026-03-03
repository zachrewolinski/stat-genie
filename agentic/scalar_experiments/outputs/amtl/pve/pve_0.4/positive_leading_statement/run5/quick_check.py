import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print('num_amtl integer-ish fraction', np.mean(np.isclose(df['num_amtl']%1, 0, atol=1e-6)))
# check if num_amtl * sockets close to integer count
prod = df['num_amtl'] * df['sockets']
print('num_amtl*sockets integer-ish fraction', np.mean(np.isclose(prod%1, 0, atol=1e-2)))
print('num_amtl min max', df['num_amtl'].min(), df['num_amtl'].max())

# check if num_amtl roughly in [0,1]
print('num_amtl in 0-1 fraction', np.mean((df['num_amtl']>=0) & (df['num_amtl']<=1)))

# check if expit(num_amtl) is close to a reasonable proportion and reconstruct counts
from scipy.special import expit
p = expit(df['num_amtl'])
# check if p*sockets close to integer
print('expit(num_amtl)*sockets integer-ish fraction', np.mean(np.isclose((p*df['sockets'])%1, 0, atol=1e-2)))

# check mean by genus for num_amtl and p
print(df.groupby('genus')['num_amtl'].mean())
print(df.groupby('genus').apply(lambda g: expit(g['num_amtl']).mean()))
