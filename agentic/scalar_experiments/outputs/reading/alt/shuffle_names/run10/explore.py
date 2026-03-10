import pandas as pd

df = pd.read_csv('reading.csv')

# basic unique values counts for key columns
cols = ['language','reader_view','device','dyslexia','dyslexia_bin','correct_rate','english_native','img_width','page_id','speed','scrolling_time']
for c in cols:
    if c in df.columns:
        print('\n', c)
        print(df[c].head())
        print('dtype', df[c].dtype)
        print('nunique', df[c].nunique(dropna=False))
        print('value_counts', df[c].value_counts(dropna=False).head(10))

# numeric columns summary
print('\nNumeric summary:')
print(df.describe(include='number').T)
