import pandas as pd

df = pd.read_csv('mortgage.csv')
for col in df.columns:
    uniq = sorted(df[col].dropna().unique())
    if len(uniq) <= 2:
        print(col, 'mean', df[col].mean(), 'counts', df[col].value_counts().to_dict())
