import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_csv('amtl.csv')
print(df.head())
print(df.describe(include='all'))
