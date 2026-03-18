import pandas as pd
import numpy as np

path = 'affairs.csv'
df = pd.read_csv(path)
print(df.head())
print(df.describe(include='all'))
print(df.dtypes)
for col in df.columns:
    print('\n', col)
    print(df[col].value_counts(dropna=False).head(10))
