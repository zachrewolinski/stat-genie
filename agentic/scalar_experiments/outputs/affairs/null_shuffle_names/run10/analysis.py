import pandas as pd

df = pd.read_csv('affairs.csv')
print(df.head())
print(df.describe(include='all'))
print('nunique:', df.nunique())
for col in df.columns:
    print('\n',col)
    print(df[col].value_counts().head(10))
