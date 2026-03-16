import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)
print('nrows', len(df))
for col in df.columns:
    if df[col].dtype == 'object':
        print(col, 'nunique', df[col].nunique(), 'sample', df[col].unique()[:5])
    else:
        print(col, 'min', df[col].min(), 'max', df[col].max(), 'mean', df[col].mean(), 'std', df[col].std())
