import pandas as pd
import numpy as np

path = 'hurricane.csv'
df = pd.read_csv(path)
print(df.head())
print('\ncolumns:', df.columns.tolist())
print('\ndtypes:\n', df.dtypes)
print('\nsummary numeric:\n', df.describe())

# show unique counts for object columns
obj_cols = df.select_dtypes(include=['object']).columns
print('\nobject cols:', obj_cols)
for c in obj_cols:
    print(c, 'nunique', df[c].nunique(), 'sample', df[c].dropna().unique()[:5])

