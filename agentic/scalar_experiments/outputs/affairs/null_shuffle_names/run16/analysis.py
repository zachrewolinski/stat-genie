import pandas as pd

# Load data

df = pd.read_csv('affairs.csv')
print(df.head())
print(df.describe(include='all'))
print('n_rows', len(df))
for col in df.columns:
    print('\n', col)
    print(df[col].dtype)
    print('unique', sorted(df[col].dropna().unique())[:20], '...')
    print('nunique', df[col].nunique())
