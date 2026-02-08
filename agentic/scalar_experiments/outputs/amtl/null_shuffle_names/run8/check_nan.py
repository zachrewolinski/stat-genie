import pandas as pd

df = pd.read_csv('amtl.csv')
print(df.isna().sum())
print('any zero num_sockets', (df['age']<=0).sum())
print('any num_missing > num_sockets', (df['genus']>df['age']).sum())
