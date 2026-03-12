import pandas as pd

df = pd.read_csv('mortgage.csv')

if 'self_employed' in df.columns and 'deny' in df.columns:
    print('self_employed == 1 - deny', (df['self_employed'] == (1-df['deny'])).mean())
if 'accept' in df.columns and 'deny' in df.columns:
    print('accept == 1 - deny', (df['accept'] == (1-df['deny'])).mean())
if 'accept' in df.columns and 'self_employed' in df.columns:
    print('accept == self_employed', (df['accept'] == df['self_employed']).mean())
