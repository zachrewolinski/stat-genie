import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
print(df[['device','dyslexia','dyslexia_bin','correct_rate']].head())

# crosstabs
print('\nDevice vs dyslexia_bin')
print(pd.crosstab(df['device'], df['dyslexia_bin'], dropna=False))

print('\nDyslexia vs dyslexia_bin')
print(pd.crosstab(df['dyslexia'], df['dyslexia_bin'], dropna=False))

print('\nDevice vs dyslexia')
print(pd.crosstab(df['device'], df['dyslexia'], dropna=False))
