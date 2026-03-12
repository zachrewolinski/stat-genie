import pandas as pd

df = pd.read_csv('mortgage.csv')
for col in ['self_employed','accept','deny','female']:
    if col in df.columns:
        print(col, df[col].mean())
