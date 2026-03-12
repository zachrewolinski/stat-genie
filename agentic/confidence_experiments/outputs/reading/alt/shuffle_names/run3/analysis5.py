import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')

for col in ['device','dyslexia','dyslexia_bin','correct_rate']:
    print(col, df[col].value_counts(dropna=False).sort_index())

# count participants by dyslexia categories using device
print('unique participants', df['speed'].nunique())
print('participants per device category')
print(df[['speed','device']].drop_duplicates()['device'].value_counts().sort_index())

print('participants per dyslexia_bin')
print(df[['speed','dyslexia_bin']].drop_duplicates()['dyslexia_bin'].value_counts().sort_index())

print('participants per correct_rate')
print(df[['speed','correct_rate']].drop_duplicates()['correct_rate'].value_counts().sort_index())

