import pandas as pd

path = 'panda_nuts.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())
print(df.dtypes)
print('nunique')
print(df.nunique())
for col in df.columns:
    if df[col].dtype == object:
        print(col, df[col].value_counts(dropna=False))
