import pandas as pd

df = pd.read_csv('soccer.csv')

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        mx = df[col].max()
        if 30 < mx <= 60:
            print(col, 'max', mx, 'min', df[col].min())
