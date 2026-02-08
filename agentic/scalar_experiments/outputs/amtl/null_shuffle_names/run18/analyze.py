import pandas as pd

df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)
for col in df.columns:
    print('\n', col)
    print(df[col].head())
    print('nunique', df[col].nunique())
    if df[col].dtype != 'object':
        print('min', df[col].min(), 'max', df[col].max())
    else:
        print('unique sample', df[col].unique()[:10])
print('rows', len(df))
