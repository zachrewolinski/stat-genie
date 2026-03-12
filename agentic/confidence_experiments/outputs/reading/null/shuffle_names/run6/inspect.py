import pandas as pd
import json

path = 'reading.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head())
print(df.dtypes)

# basic stats for numeric columns
num = df.select_dtypes(include='number')
print('numeric columns', num.columns.tolist())
print(num.describe().T[['count','mean','std','min','max']].head(20))

# unique counts for low-card columns
for col in df.columns:
    nun = df[col].nunique(dropna=False)
    if nun<=10:
        print('col', col, 'nunique', nun, 'values', df[col].unique()[:10])
