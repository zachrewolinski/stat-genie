import pandas as pd

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

path = 'reading.csv'
df = pd.read_csv(path)

for col in df.columns:
    print('\n===', col, '===')
    print('dtype', df[col].dtype)
    if df[col].dtype == 'object':
        print('unique', df[col].nunique())
        print(df[col].value_counts().head(10))
    else:
        print('min', df[col].min(), 'max', df[col].max(), 'mean', df[col].mean())
        print('unique', df[col].nunique())
        print(df[col].value_counts().head(10))
