import pandas as pd
import numpy as np

path = 'soccer.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)

# numeric columns summary
num_cols = df.select_dtypes(include=[np.number]).columns
print('numeric columns', list(num_cols))
print(df[num_cols].describe().T[['count','mean','std','min','max']])

# check potential games/red cards columns by value ranges
for col in num_cols:
    vals = df[col]
    if vals.max() <= 50:
        print(col, 'max', vals.max(), 'mean', vals.mean(), 'unique sample', vals.dropna().unique()[:10])

