import pandas as pd

_df=pd.read_csv('amtl.csv')
print(_df.head())
print('\nsummary')
print(_df.describe(include='all'))

for col in _df.columns:
    print('\n',col)
    if _df[col].dtype=='object':
        print(_df[col].value_counts().head())
    else:
        print(_df[col].min(), _df[col].max(), _df[col].mean())

