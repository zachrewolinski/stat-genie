import pandas as pd
import numpy as np


df = pd.read_csv('amtl.csv')
print(df.head())
print('\nColumns:', df.columns.tolist())
print('\nDtypes:', df.dtypes)

# Summaries
for col in df.columns:
    if df[col].dtype == 'object':
        print('\n', col, 'nunique', df[col].nunique())
        print(df[col].value_counts().head())
    else:
        print('\n', col, 'min', df[col].min(), 'max', df[col].max(), 'mean', df[col].mean())
        print(df[col].describe())

