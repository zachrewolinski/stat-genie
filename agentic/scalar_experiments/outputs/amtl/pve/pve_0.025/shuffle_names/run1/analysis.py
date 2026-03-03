import pandas as pd
import numpy as np

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())
print('\nColumns and dtypes:')
print(df.dtypes)

# Unique values for categorical columns
for col in df.columns:
    if df[col].dtype == 'object':
        print('\n', col, 'unique', df[col].nunique())
        print(df[col].value_counts().head())

# Summary stats for numeric
print('\nNumeric summary:')
print(df.describe())

# Check for missing
print('\nMissing:')
print(df.isna().sum())

# Explore potential outcome relationship
# Inspect distributions of numeric columns
for col in df.columns:
    if df[col].dtype != 'object':
        print('\n', col, 'min', df[col].min(), 'max', df[col].max(), 'mean', df[col].mean())

# Try to infer which columns are counts by checking near-integers
for col in df.columns:
    if df[col].dtype != 'object':
        frac = np.mean(np.isclose(df[col], np.round(df[col]), atol=1e-6))
        print(col, 'fraction integer-like', frac)

