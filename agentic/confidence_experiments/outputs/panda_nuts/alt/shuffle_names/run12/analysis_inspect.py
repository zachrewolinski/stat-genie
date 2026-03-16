import pandas as pd

path = 'panda_nuts.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))
print('unique values')
for col in df.columns:
    print(col, df[col].nunique(), df[col].unique()[:10])
