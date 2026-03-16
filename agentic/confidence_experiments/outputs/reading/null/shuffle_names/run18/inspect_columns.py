import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 200)

path = 'reading.csv'
df = pd.read_csv(path)

print('Missing counts:')
print(df.isna().sum())

# For categorical/object columns, show value counts top
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"\n{col} value counts:")
        print(df[col].value_counts(dropna=False).head(10))

# For numeric columns with few unique values
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 10:
        print(f"\n{col} unique values: {sorted(df[col].dropna().unique())}")
