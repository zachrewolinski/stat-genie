import pandas as pd

df = pd.read_csv('amtl.csv')
ct = pd.crosstab(df['sockets'], df['age'])
print(ct)
