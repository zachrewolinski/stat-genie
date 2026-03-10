import pandas as pd

df = pd.read_csv('panda_nuts.csv')

for col in df.columns:
    print('\n', col)
    print('dtype', df[col].dtype)
    print('nunique', df[col].nunique())
    print('sample unique', df[col].unique()[:20])
