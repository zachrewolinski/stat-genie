import pandas as pd
import numpy as np


df = pd.read_csv('affairs.csv')
print(df.head())
print('\ncolumns:', df.columns.tolist())
print('\nunique counts:')
for col in df.columns:
    print(col, df[col].nunique())
print('\nsummary:')
print(df.describe(include='all'))
