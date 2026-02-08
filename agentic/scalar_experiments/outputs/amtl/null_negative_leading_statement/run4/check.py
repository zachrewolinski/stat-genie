import pandas as pd

df = pd.read_csv('amtl.csv')
print('rows', len(df))
print('sockets min', df['sockets'].min())
print('num_amtl max', df['num_amtl'].max())
print('num_amtl > sockets', (df['num_amtl'] > df['sockets']).sum())
print('num_amtl / sockets max', (df['num_amtl'] / df['sockets']).max())
print('num_amtl / sockets min', (df['num_amtl'] / df['sockets']).min())
print('any sockets <=0', (df['sockets'] <= 0).sum())
print('any missing in key', df[['num_amtl','sockets','age','prob_male','tooth_class','genus']].isna().any())
