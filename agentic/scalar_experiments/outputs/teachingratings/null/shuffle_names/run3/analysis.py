import pandas as pd
import numpy as np

df = pd.read_csv('teachingratings.csv')
print(df.head())
print('\nDtypes:')
print(df.dtypes)
print('\nSummary:')
print(df.describe(include='all'))

# unique counts for object columns
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"\n{col} unique values ({df[col].nunique()}):", df[col].unique())

# numeric ranges
for col in df.columns:
    if df[col].dtype != 'object':
        print(f"\n{col} min/max: {df[col].min()} / {df[col].max()} unique={df[col].nunique()}")
