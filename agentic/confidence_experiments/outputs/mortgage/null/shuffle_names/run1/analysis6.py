import pandas as pd

df = pd.read_csv('mortgage.csv')
print('accept vs self_employed')
print(pd.crosstab(df['accept'], df['self_employed']))
print('accept vs denied_PMI')
print(pd.crosstab(df['accept'], df['denied_PMI']))
print('self_employed mean', df['self_employed'].mean(), 'accept mean', df['accept'].mean())
