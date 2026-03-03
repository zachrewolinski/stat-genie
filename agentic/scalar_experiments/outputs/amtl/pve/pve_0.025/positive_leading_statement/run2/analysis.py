import pandas as pd

df = pd.read_csv("amtl.csv")
print(df.head())
print(df.dtypes)
print(df.describe(include="all"))
