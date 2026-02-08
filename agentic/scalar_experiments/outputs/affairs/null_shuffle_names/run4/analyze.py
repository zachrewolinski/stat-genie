import pandas as pd

# Load data

df = pd.read_csv('affairs.csv')
print(df.head())
print(df.describe(include='all'))
print(df.dtypes)
print('columns', df.columns.tolist())

for col in df.columns:
    print('\n', col)
    print(df[col].head(10).tolist())
    print('unique', df[col].nunique())
    if df[col].dtype != object:
        print('min/max', df[col].min(), df[col].max())
    else:
        print(df[col].value_counts().head())
