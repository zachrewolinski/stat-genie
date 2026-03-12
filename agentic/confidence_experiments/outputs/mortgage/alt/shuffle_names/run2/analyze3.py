import pandas as pd

df = pd.read_csv('mortgage.csv')
for c in df.columns:
    if df[c].dropna().nunique()<=2:
        print(c, df[c].value_counts().to_dict())
