import pandas as pd
import numpy as np

# Load data

df = pd.read_csv('affairs.csv')
print(df.head())
print(df.describe(include='all'))
print('unique values per column:')
for col in df.columns:
    print(col, df[col].unique()[:20], 'n=', df[col].nunique())
