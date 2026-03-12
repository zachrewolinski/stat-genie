import pandas as pd
import numpy as np

df = pd.read_csv('mortgage.csv')
print(df.head())
print('columns', df.columns.tolist())

for col in ['accept','deny','female']:
    if col in df.columns:
        print('\n', col)
        print(df[col].value_counts(dropna=False).sort_index())

if 'accept' in df.columns and 'deny' in df.columns:
    s = df['accept'] + df['deny']
    print('\naccept+deny unique', np.unique(s)[:10])
    print('accept+deny value counts', s.value_counts().head())
    print('corr accept/deny', df['accept'].corr(df['deny']))

if 'deny' in df.columns:
    print('deny mean', df['deny'].mean())
if 'accept' in df.columns:
    print('accept mean', df['accept'].mean())

