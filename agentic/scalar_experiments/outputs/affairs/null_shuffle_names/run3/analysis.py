import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')
print(df.head())
print(df.describe(include='all'))

for col in df.columns:
    print('\n', col)
    print('dtype', df[col].dtype)
    print('unique', df[col].nunique())
    print('min', df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else '')
    print('max', df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else '')
    print('value_counts head', df[col].value_counts().head())
