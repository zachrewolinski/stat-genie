import pandas as pd

df = pd.read_csv('amtl.csv')
print('corr num_amtl vs sockets', df['num_amtl'].corr(df['sockets']))
print('corr num_amtl vs age', df['num_amtl'].corr(df['age']))
