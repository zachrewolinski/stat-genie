import pandas as pd
import numpy as np


df = pd.read_csv('affairs.csv')
print('columns', df.columns.tolist())
print(df.head())
print(df.describe(include='all'))

# compute unique values counts for each column
for col in df.columns:
    print('\n', col)
    print('dtype', df[col].dtype)
    print('nunique', df[col].nunique(dropna=False))
    print('unique sample', df[col].unique()[:20])
    print(df[col].value_counts(dropna=False).head(10))

