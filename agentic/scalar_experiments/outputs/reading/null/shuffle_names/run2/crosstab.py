import pandas as pd

df = pd.read_csv('reading.csv')

for bin_col in ['dyslexia_bin','correct_rate']:
    print('\n', bin_col)
    ct = pd.crosstab(df['device'], df[bin_col])
    print(ct)

    ct2 = pd.crosstab(df['dyslexia'], df[bin_col])
    print('vs dyslexia col')
    print(ct2)
