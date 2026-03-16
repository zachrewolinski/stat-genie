import pandas as pd

path = "soccer.csv"
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.describe(include='all').transpose().head(15))
print("columns", df.columns.tolist())
