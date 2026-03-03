import pandas as pd
import json

# Load data

df = pd.read_csv('amtl.csv')
print(df.head())
print('\nDtypes:')
print(df.dtypes)
print('\nUnique counts:')
print(df.nunique())

# show unique values for categorical columns (limited)
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"\n{col} unique sample:", df[col].unique()[:10])

# describe numeric
print('\nNumeric describe:')
print(df.describe())

# check missing values
print('\nMissing values:')
print(df.isna().sum())
