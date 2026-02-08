import pandas as pd
import numpy as np

path = 'amtl.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
for col in df.columns:
    print('\n', col)
    print(df[col].head(10))
    if df[col].dtype == 'object':
        print('nunique', df[col].nunique())
        print('unique sample', df[col].unique()[:10])
    else:
        print('describe', df[col].describe())
