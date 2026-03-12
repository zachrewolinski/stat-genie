import pandas as pd

path = 'mortgage.csv'

df = pd.read_csv(path)
print(df.head())
print(df.describe(include='all'))
print(df.columns)
for col in df.columns:
    if df[col].dropna().nunique() <= 5:
        print(col, df[col].value_counts(dropna=False).head())
