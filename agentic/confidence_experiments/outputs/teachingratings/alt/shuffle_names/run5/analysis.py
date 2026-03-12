import pandas as pd
import json

csv_path = 'teachingratings.csv'
df = pd.read_csv(csv_path)
print(df.head())
print(df.columns)
print(df.describe(include='all'))
