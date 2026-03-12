import pandas as pd

path = 'panda_nuts.csv'
df = pd.read_csv(path)
print(df.head())
print('\ninfo')
print(df.dtypes)
print('\nunique counts')
for col in df.columns:
    print(col, df[col].nunique(), df[col].head(10).tolist())
print('\nsummary')
print(df.describe(include='all'))
