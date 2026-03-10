import pandas as pd

path = "panda_nuts.csv"
df = pd.read_csv(path)
print(df.head())
print(df.columns)
print(df.shape)
print(df.dtypes)
