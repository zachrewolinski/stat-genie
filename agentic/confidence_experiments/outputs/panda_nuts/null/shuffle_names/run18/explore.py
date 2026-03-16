import pandas as pd

pd.set_option('display.width', 120)

df = pd.read_csv('panda_nuts.csv')
print(df.head())
print('\nColumns:', df.columns.tolist())
for col in df.columns:
    print('\n', col)
    print(df[col].dtype)
    print('unique sample:', pd.unique(df[col])[:10])
    print('nunique:', df[col].nunique())
    if pd.api.types.is_numeric_dtype(df[col]):
        print('min/max/mean:', df[col].min(), df[col].max(), df[col].mean())
