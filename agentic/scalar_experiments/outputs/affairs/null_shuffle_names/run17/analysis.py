import pandas as pd
import json

# Load data
csv_path = 'affairs.csv'
df = pd.read_csv(csv_path)
print('Columns:', df.columns.tolist())
print('Shape:', df.shape)
print(df.head())
print('\nDtypes:')
print(df.dtypes)

# Basic stats for key columns
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        print('\n', col, df[col].describe())
    else:
        print('\n', col, df[col].value_counts().head())
