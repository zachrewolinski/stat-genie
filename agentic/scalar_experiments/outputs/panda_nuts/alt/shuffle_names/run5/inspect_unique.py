import pandas as pd

df = pd.read_csv('panda_nuts.csv')
for col in df.columns:
    print(col, df[col].nunique(), df[col].unique()[:10])
