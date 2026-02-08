import pandas as pd

path = 'affairs.csv'
df = pd.read_csv(path)
print(df.head())
print(df.columns)
print(df.shape)
print(df['children'].value_counts(dropna=False))
print(df['affairs'].describe())
