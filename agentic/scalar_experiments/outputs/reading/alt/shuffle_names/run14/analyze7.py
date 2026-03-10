import pandas as pd

df = pd.read_csv('reading.csv')
print('dyslexia counts')
print(df['dyslexia'].value_counts())
print('device counts')
print(df['device'].value_counts())
