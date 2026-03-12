import pandas as pd

path='reading.csv'
df=pd.read_csv(path)

# cross-tab device vs correct_rate and dyslexia_bin
print('device value counts')
print(df['device'].value_counts(dropna=False))
print('correct_rate value counts')
print(df['correct_rate'].value_counts(dropna=False))
print('dyslexia_bin value counts')
print(df['dyslexia_bin'].value_counts(dropna=False))

print('\nCrosstab device vs correct_rate (dropna)')
print(pd.crosstab(df['device'], df['correct_rate']))

print('\nCrosstab device vs dyslexia_bin (dropna)')
print(pd.crosstab(df['device'], df['dyslexia_bin']))
