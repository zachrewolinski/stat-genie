import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
for col in ['num_amtl','age','stdev_age','prob_male']:
    s=df[col]
    print(col, 'mean', s.mean(), 'std', s.std(), 'min', s.min(), 'max', s.max())

# check if num_amtl close to proportion of sockets
print('num_amtl * sockets stats', (df['num_amtl']*df['sockets']).describe())
print('num_amtl / sockets stats', (df['num_amtl']/df['sockets']).describe())

# check if maybe standardized by within tooth class or something? compute by genus
print('num_amtl by genus mean')
print(df.groupby('genus')['num_amtl'].mean())
print('num_amtl by genus std')
print(df.groupby('genus')['num_amtl'].std())

# maybe num_amtl equals logit proportion? try transform to proportion via logistic
logit = df['num_amtl']
prop = 1/(1+np.exp(-logit))
print('prop from logit min max', prop.min(), prop.max())
print('prop from logit summary', prop.describe())

