import pandas as pd

_df=pd.read_csv('mortgage.csv')
print('self_employed mean', _df['self_employed'].mean())
print('deny mean', _df['deny'].mean())
print(pd.crosstab(_df['self_employed'], _df['deny']))
print('corr', _df['self_employed'].corr(_df['deny']))
print('agreement rate', ((1-_df['self_employed'])==_df['deny']).mean())
