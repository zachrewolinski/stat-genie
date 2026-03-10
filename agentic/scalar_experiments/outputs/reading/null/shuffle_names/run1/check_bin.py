import pandas as pd

df = pd.read_csv('reading.csv')
# compare dyslexia_bin with dyslexia
print('dyslexia_bin=1 counts by dyslexia:')
print(df[df['dyslexia_bin']==1]['dyslexia'].value_counts())
print('dyslexia_bin=1 counts by device:')
print(df[df['dyslexia_bin']==1]['device'].value_counts())

# proportion of dyslexia_bin=1 within dyslexia categories
print('\nproportion dyslexia_bin=1 within dyslexia categories:')
print(df.groupby('dyslexia')['dyslexia_bin'].mean())
print('\nproportion dyslexia_bin=1 within device categories:')
print(df.groupby('device')['dyslexia_bin'].mean())
