import pandas as pd

_df=pd.read_csv('mortgage.csv')
print('accept mean', _df['accept'].mean())
print('deny mean', _df['deny'].mean())
print('accept vs deny crosstab')
print(pd.crosstab(_df['accept'], _df['deny']))
print('corr', _df['accept'].corr(_df['deny']))

