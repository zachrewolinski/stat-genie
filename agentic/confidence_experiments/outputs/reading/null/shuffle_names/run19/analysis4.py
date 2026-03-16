import pandas as pd

df = pd.read_csv('reading.csv')
print(pd.crosstab(df['device'], df['correct_rate'], dropna=False))
print('\n')
print(pd.crosstab(df['dyslexia'], df['correct_rate'], dropna=False))

