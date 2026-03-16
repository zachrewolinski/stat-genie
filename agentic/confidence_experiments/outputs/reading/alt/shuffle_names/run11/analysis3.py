import pandas as pd

path = 'reading.csv'
df = pd.read_csv(path)

for c in ['device','dyslexia']:
    print('\n', c)
    print(df[c].value_counts(dropna=False))

# check proportion of dyslexia_bin column
print('\n dyslexia_bin')
print(df['dyslexia_bin'].value_counts(dropna=False))
