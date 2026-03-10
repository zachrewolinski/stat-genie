import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

# cross tabs between potential dyslexia indicators
print(pd.crosstab(df['device'], df['correct_rate'], dropna=False))
print(pd.crosstab(df['device'], df['dyslexia_bin'], dropna=False))
print(pd.crosstab(df['dyslexia'], df['correct_rate'], dropna=False))
print(pd.crosstab(df['dyslexia'], df['dyslexia_bin'], dropna=False))

# look at unique participants cross-tab
uniq = df[['speed','device','correct_rate','dyslexia_bin','dyslexia']].drop_duplicates()
print('participants device vs correct_rate')
print(pd.crosstab(uniq['device'], uniq['correct_rate'], dropna=False))
print('participants device vs dyslexia_bin')
print(pd.crosstab(uniq['device'], uniq['dyslexia_bin'], dropna=False))

