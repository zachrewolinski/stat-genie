import pandas as pd

df=pd.read_csv('reading.csv')

# binary numeric columns
for col in ['language','dyslexia_bin','correct_rate']:
    print(col, df[col].value_counts(dropna=False))

# Y/N column
print('img_width', df['img_width'].value_counts(dropna=False))

# dyslexia and device
print('dyslexia', df['dyslexia'].value_counts(dropna=False))
print('device', df['device'].value_counts(dropna=False))

# reader_view column (languages)
print('reader_view unique', df['reader_view'].value_counts().head(10))

