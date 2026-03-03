import pandas as pd
import numpy as np


df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)
for col in df.columns:
    print('\n', col)
    if df[col].dtype == object:
        print('nunique', df[col].nunique())
        print(df[col].unique()[:10])
    else:
        print(df[col].describe())
