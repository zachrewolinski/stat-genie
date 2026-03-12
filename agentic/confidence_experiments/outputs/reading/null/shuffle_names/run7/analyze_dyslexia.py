import pandas as pd


df = pd.read_csv('reading.csv')

for col in ['device','dyslexia','dyslexia_bin','correct_rate']:
    print(col, df[col].value_counts(dropna=False))

# check relation between dyslexia_bin and device/dyslexia
print('\nP(dyslexia_bin=1 | device)')
print(df.groupby('device')['dyslexia_bin'].mean())

print('\nP(dyslexia_bin=1 | dyslexia)')
print(df.groupby('dyslexia')['dyslexia_bin'].mean())

