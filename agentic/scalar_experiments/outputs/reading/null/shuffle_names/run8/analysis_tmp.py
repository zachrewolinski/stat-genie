import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)
print(df.head())
print('\nColumns:', df.columns.tolist())
print('\nDtypes:')
print(df.dtypes)
print('\nSummary:')
print(df.describe(include='all').T.head(20))
