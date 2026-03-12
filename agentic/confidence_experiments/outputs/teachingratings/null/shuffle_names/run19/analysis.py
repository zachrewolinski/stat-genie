import pandas as pd

# Load data

df = pd.read_csv('teachingratings.csv')
print('columns:', list(df.columns))
print(df.head())
print(df.describe(include='all').transpose())
