import pandas as pd
import numpy as np

path = 'affairs.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.nunique())
# show unique values for categorical columns
for col in df.columns:
    if df[col].dtype == object:
        print(col, df[col].unique()[:10])
    else:
        # show min/max and few unique
        print(col, df[col].min(), df[col].max(), df[col].unique()[:10])
