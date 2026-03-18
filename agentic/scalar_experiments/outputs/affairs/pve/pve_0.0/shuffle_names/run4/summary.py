import pandas as pd

df = pd.read_csv('affairs.csv')
print(df.describe(include='all').transpose())
