import pandas as pd

df = pd.read_csv('reading.csv')

for col in df.columns:
    if df[col].dtype == 'object':
        print(col, df[col].unique()[:10], 'nunique', df[col].nunique())
    else:
        nun = df[col].nunique()
        if nun <= 10:
            print(col, sorted(df[col].dropna().unique()), 'nunique', nun)
