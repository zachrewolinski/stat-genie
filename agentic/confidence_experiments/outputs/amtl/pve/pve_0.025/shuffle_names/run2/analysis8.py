import pandas as pd

df = pd.read_csv('amtl.csv')
print('genus mean', df['genus'].mean(), 'std', df['genus'].std())
print('genus by sockets mean', df.groupby('sockets')['genus'].mean())
print('genus min/max by sockets', df.groupby('sockets')['genus'].agg(['min','max']))
