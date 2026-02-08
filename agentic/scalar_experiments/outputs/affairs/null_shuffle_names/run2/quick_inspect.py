import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')
print(df.head())
for col in df.columns:
    vals = df[col].dropna().unique()
    # show sorted unique if numeric and small
    if pd.api.types.is_numeric_dtype(df[col]):
        uniq = np.unique(df[col])
        if len(uniq) <= 15:
            print(col, 'unique', uniq)
        else:
            print(col, 'min', df[col].min(), 'max', df[col].max(), 'nunique', df[col].nunique())
    else:
        print(col, 'unique', pd.unique(df[col]))
