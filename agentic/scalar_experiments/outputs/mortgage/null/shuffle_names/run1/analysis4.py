import pandas as pd

df = pd.read_csv('mortgage.csv')
print((df['self_employed'] + df['deny']).value_counts().sort_index())
print('rows where !=1', (df['self_employed'] + df['deny'] != 1).sum())
print(pd.crosstab(df['self_employed'], df['deny']))
