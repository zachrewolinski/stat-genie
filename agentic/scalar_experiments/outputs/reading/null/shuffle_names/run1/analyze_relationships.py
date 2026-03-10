import pandas as pd

pd.set_option('display.max_rows', 200)

df = pd.read_csv('reading.csv')

# cross tabs
for col in ['device', 'dyslexia']:
    ct = pd.crosstab(df[col], df['dyslexia_bin'])
    print('\nCrosstab', col, 'vs dyslexia_bin')
    print(ct)

for col in ['device', 'dyslexia']:
    ct = pd.crosstab(df[col], df['correct_rate'])
    print('\nCrosstab', col, 'vs correct_rate')
    print(ct)
