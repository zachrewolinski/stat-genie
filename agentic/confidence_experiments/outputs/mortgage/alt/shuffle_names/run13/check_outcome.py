import pandas as pd
import numpy as np

df = pd.read_csv('mortgage.csv')
if 'self_employed' in df.columns and 'deny' in df.columns:
    s = df['self_employed'] + df['deny']
    print('self_employed+deny unique', np.unique(s)[:10])
    print('value counts', s.value_counts().head())
    print('corr', df['self_employed'].corr(df['deny']))

print('self_employed mean', df['self_employed'].mean())
print('deny mean', df['deny'].mean())
