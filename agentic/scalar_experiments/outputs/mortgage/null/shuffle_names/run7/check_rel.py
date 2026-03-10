import pandas as pd

df = pd.read_csv('mortgage.csv')
print('corr deny vs self_employed', df['deny'].corr(df['self_employed']))
print('mean deny', df['deny'].mean(), 'mean self_employed', df['self_employed'].mean())
print('cross tab')
print(pd.crosstab(df['deny'], df['self_employed']))

