import pandas as pd

df = pd.read_csv('reading.csv')

ct = pd.crosstab(df['device'], df['dyslexia_bin'], dropna=False)
print(ct)

ct2 = pd.crosstab(df['dyslexia'], df['dyslexia_bin'], dropna=False)
print('\n', ct2)
