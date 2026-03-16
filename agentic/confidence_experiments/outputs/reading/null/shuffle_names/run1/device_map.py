import pandas as pd

df = pd.read_csv('reading.csv')
ct = pd.crosstab(df['english_native'], df['device'])
print(ct)
