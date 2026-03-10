import pandas as pd
import json

path = 'crofoot.csv'
df = pd.read_csv(path)
print(df.head())
print(df.describe(include='all'))
print(df.columns)
