import pandas as pd

_df = pd.read_csv('soccer.csv')

for col in _df.columns:
    if _df[col].nunique() == 5:
        print(col, _df[col].dtype, sorted(_df[col].dropna().unique()))

