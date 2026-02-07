import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 50)

df = pd.read_csv('amtl.csv')
print('columns', df.columns.tolist())
print('\nhead')
print(df.head())

for col in df.columns:
    print('\n', col)
    print(df[col].dtype)
    if df[col].dtype == 'object':
        print('nunique', df[col].nunique())
        print(df[col].value_counts().head(10))
    else:
        print(df[col].describe())

# check counts and missing
print('\nmissing counts')
print(df.isna().sum())

# check candidate mappings
# correlations between numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns
print('\nnum cols', num_cols.tolist())
print('\ncorrelation')
print(df[num_cols].corr())

