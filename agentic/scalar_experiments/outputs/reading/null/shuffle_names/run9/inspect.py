import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print('shape', df.shape)

# summary of unique counts
for col in df.columns:
    nunique = df[col].nunique(dropna=False)
    print(col, 'nunique', nunique, 'dtype', df[col].dtype)

