import pandas as pd

df = pd.read_csv('mortgage.csv')
print((df['self_employed'] + df['deny']).unique())
print(((df['self_employed'] + df['deny'])==1).mean())
print(((df['self_employed'] + df['deny'])==0).sum(), ((df['self_employed'] + df['deny'])==2).sum())
