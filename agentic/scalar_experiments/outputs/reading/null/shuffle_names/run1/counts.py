import pandas as pd

df = pd.read_csv('reading.csv')
print('dyslexia counts:', df['dyslexia'].value_counts())
print('device counts:', df['device'].value_counts())
print('dyslexia_bin counts:', df['dyslexia_bin'].value_counts())
print('correct_rate counts:', df['correct_rate'].value_counts())
