import pandas as pd

df = pd.read_csv('amtl.csv')
print(df.groupby('genus')['num_amtl'].mean())
print(df.groupby('genus')['num_amtl'].describe())
