import pandas as pd

df = pd.read_csv('mortgage.csv')
print(pd.crosstab(df['self_employed'], df['deny']))
