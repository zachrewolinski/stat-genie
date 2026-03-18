import pandas as pd
import numpy as np

col=pd.read_csv('affairs.csv')['age']
print(col.describe())
print('skew', col.skew(), 'kurt', col.kurt())
# show proportion below 0
print('pct<0', (col<0).mean())
print('quantiles', col.quantile([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]))
