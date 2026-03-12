import pandas as pd
import numpy as np

# Load data
file_path = 'amtl.csv'
df = pd.read_csv(file_path)

print('Columns:', df.columns.tolist())
print('\nHead:')
print(df.head())

# Inspect dtypes
print('\nDtypes:')
print(df.dtypes)

# Count unique values for categorical-ish columns
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"\n{col} unique count: {df[col].nunique()}")
        print(f"Sample: {df[col].unique()[:5]}")

# Summary stats for numeric columns
print('\nNumeric summary:')
print(df.describe())

# Check if any numeric columns are integer-like
for col in df.columns:
    if np.issubdtype(df[col].dtype, np.number):
        # check integer-like proportion
        vals = df[col].dropna()
        frac_int = np.mean(np.isclose(vals, np.round(vals)))
        print(f"{col}: fraction integer-like = {frac_int:.3f}")

# Check groupings for potential mapping
print('\nValue counts for tooth_class:')
print(df['tooth_class'].value_counts())
print('\nValue counts for sockets:')
print(df['sockets'].value_counts())
print('\nValue counts for specimen:')
print(df['specimen'].value_counts().head())

