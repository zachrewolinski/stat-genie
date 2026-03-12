import pandas as pd

df = pd.read_csv('amtl.csv')
print(df.groupby('sockets')['genus'].mean())
print(df.groupby('sockets')['genus'].agg(['mean','std','min','max']))
