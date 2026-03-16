import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print('pop range', df['pop'].min(), df['pop'].max())
print('num_amtl range', df['num_amtl'].min(), df['num_amtl'].max())
print('pop-num_amtl mean diff', (df['pop']-df['num_amtl']).mean())
print('pop/num_amtl mean ratio', (df['pop']/df['num_amtl']).mean())
# maybe num_amtl is sqrt(pop)
print('corr num_amtl vs sqrt(pop)', df['num_amtl'].corr(np.sqrt(df['pop'])))
print('corr num_amtl vs log(pop)', df['num_amtl'].corr(np.log(df['pop'])))
