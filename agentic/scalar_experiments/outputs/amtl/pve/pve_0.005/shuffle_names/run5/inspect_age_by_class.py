import pandas as pd

df = pd.read_csv('amtl.csv')
print(df.groupby('sockets')['age'].describe())
