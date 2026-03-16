import pandas as pd

path = 'reading.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head())
print(df.dtypes)
for col in df.columns:
    print('\n', col)
    print('nunique', df[col].nunique(dropna=False))
    print('sample', df[col].dropna().astype(str).head(5).tolist())
