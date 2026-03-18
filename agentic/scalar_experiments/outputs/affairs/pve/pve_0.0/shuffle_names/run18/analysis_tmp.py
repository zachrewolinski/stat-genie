import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')
print('columns', df.columns.tolist())
for col in df.columns:
    print('\n', col)
    print(df[col].dtype)
    print('nunique', df[col].nunique())
    if df[col].dtype == object:
        print(df[col].value_counts().head())
    else:
        print(df[col].describe())
        print('unique sample', sorted(df[col].unique())[:15])
