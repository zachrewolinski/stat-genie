import pandas as pd

df = pd.read_csv('reading.csv')
print('dyslexia_bin counts', df['dyslexia_bin'].value_counts())
print('device vs dyslexia_bin crosstab')
print(pd.crosstab(df['device'], df['dyslexia_bin']))
print('dyslexia vs dyslexia_bin crosstab')
print(pd.crosstab(df['dyslexia'], df['dyslexia_bin']))
