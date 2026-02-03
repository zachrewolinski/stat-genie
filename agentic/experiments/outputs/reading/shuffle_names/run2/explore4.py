import pandas as pd

df = pd.read_csv('reading.csv')
print('device counts', df['device'].value_counts(dropna=False))
print('dyslexia counts', df['dyslexia'].value_counts(dropna=False))
