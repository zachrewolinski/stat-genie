import pandas as pd

df = pd.read_csv('amtl.csv')

print(df.groupby('sockets')['age'].describe())
print('\nmax age by sockets')
print(df.groupby('sockets')['age'].max())

print('\nmean age by sockets')
print(df.groupby('sockets')['age'].mean())
