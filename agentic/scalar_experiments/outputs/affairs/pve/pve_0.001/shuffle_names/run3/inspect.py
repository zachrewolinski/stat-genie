import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')
print(df.head())
print('\nSummary:')
print(df.describe(include='all').T)

# unique counts
print('\nUnique counts:')
print(df.nunique())

# show unique values for any columns with <=10 unique
for col in df.columns:
    uniq = df[col].unique()
    if len(uniq) <= 10:
        print(f"\n{col} unique values ({len(uniq)}): {sorted(uniq)}")
