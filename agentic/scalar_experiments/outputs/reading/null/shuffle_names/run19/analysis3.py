import pandas as pd


df = pd.read_csv('reading.csv')
missing = df.isna().sum().sort_values(ascending=False)
print(missing)

for col in ['language','dyslexia_bin','correct_rate','device','dyslexia']:
    print('\n', col, df[col].value_counts(dropna=False))

# check relation between language (binary) and reader_view text column
print('\nLanguage vs reader_view unique:')
print(pd.crosstab(df['language'], df['reader_view'].fillna('NA')))

# check relation between device and english_native
print('\nDevice vs english_native:')
print(pd.crosstab(df['device'], df['english_native'].fillna('NA')))

# check relation between dyslexia_bin and dyslexia
print('\nDyslexia vs dyslexia_bin:')
print(pd.crosstab(df['dyslexia'], df['dyslexia_bin']))
