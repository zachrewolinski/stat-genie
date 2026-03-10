import pandas as pd

path = 'teachingratings.csv'
df = pd.read_csv(path)
print(df.head())
print(df.columns.tolist())
print(df.dtypes)
print(df.shape)
