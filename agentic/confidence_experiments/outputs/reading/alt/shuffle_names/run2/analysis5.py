import pandas as pd

df = pd.read_csv('reading.csv')
print(pd.crosstab(df['dyslexia'], df['dyslexia_bin'], dropna=False))
print(pd.crosstab(df['dyslexia'], df['correct_rate'], dropna=False))
