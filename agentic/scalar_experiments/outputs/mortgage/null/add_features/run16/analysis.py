import pandas as pd
import numpy as np

# Load data

df = pd.read_csv('mortgage.csv')

print('shape', df.shape)
print('columns', df.columns.tolist())

# Basic missingness
missing = df.isna().mean().sort_values(ascending=False)
print('missing_top', missing.head(10))

# Check key vars
for col in ['female','accept','deny']:
    if col in df.columns:
        print(col, df[col].describe())

# Check if accept + deny are complements
if 'accept' in df.columns and 'deny' in df.columns:
    comp = (df['accept'] + df['deny']).value_counts(dropna=False)
    print('accept+deny value counts', comp.head())

# Simple crosstab
if 'female' in df.columns and 'accept' in df.columns:
    ct = pd.crosstab(df['female'], df['accept'], dropna=False)
    print('crosstab female x accept:\n', ct)

