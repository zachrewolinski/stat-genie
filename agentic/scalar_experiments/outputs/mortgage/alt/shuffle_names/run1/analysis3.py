import pandas as pd

_df = pd.read_csv('mortgage.csv')

print('self_employed + deny describe')
print((_df['self_employed'] + _df['deny']).describe())
print('self_employed == 1 - deny', (_df['self_employed'] == 1 - _df['deny']).mean())

