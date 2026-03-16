import pandas as pd

df = pd.read_csv('amtl.csv')
print(df.groupby('sockets')['genus'].min())
print(df.groupby('sockets')['genus'].max())

