import pandas as pd

path = 'hurricane.csv'

df = pd.read_csv(path)
print(df.head())
print(df.describe(include='all'))
