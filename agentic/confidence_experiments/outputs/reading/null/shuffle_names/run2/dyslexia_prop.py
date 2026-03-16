import pandas as pd

df = pd.read_csv('reading.csv')
print(df['device'].value_counts(dropna=False).sort_index())
print('prop device>0', (df['device']>0).mean())
print('dyslexia_bin mean', df['dyslexia_bin'].mean())
print('correct_rate mean', df['correct_rate'].mean())
