import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')
print(df.head())
print(df.dtypes)
print(df.nunique())
for col in df.columns:
    if df[col].dtype == object:
        print(col, df[col].unique()[:10])
    else:
        print(col, df[col].min(), df[col].max(), df[col].unique()[:10])
