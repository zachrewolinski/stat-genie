import pandas as pd

col=pd.read_csv('affairs.csv')['education']
print(col.describe())
print('skew', col.skew(), 'kurt', col.kurt())
print('pct<0', (col<0).mean())
