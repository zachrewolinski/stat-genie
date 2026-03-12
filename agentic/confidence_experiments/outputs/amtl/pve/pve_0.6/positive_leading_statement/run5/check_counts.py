import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
prod = df['num_amtl'] * df['sockets']
close_int = (np.abs(prod - np.round(prod)) < 0.05)
print('close to integer proportion', close_int.mean())
print('min max prod', prod.min(), prod.max())
print('num_amtl >1 proportion', (df['num_amtl']>1).mean())
print('num_amtl <0 proportion', (df['num_amtl']<0).mean())

# show some rows where num_amtl>1
print(df[df['num_amtl']>1].head())

# show some rows where num_amtl<0
print(df[df['num_amtl']<0].head())
