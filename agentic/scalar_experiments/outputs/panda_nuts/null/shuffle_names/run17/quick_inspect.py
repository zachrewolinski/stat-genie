import pandas as pd

df = pd.read_csv('panda_nuts.csv')
print(df.head())
print(df.dtypes)
for col in df.columns:
    print('\n', col)
    print(df[col].unique()[:20])
    print(df[col].describe(include='all'))
