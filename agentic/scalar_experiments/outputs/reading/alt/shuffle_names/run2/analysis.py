import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df.isna().mean())
print(df['dyslexia'].value_counts(dropna=False).head())
print(df['dyslexia_bin'].value_counts(dropna=False))
print(df['language'].value_counts(dropna=False).head())
print(df['reader_view'].value_counts(dropna=False).head())
print(df['english_native'].value_counts(dropna=False).head())

# check ranges for time columns
for col in ['running_time','adjusted_running_time','age','gender','scrolling_time']:
    if col in df.columns:
        print(col, df[col].min(), df[col].max(), df[col].median())

# compute reading speed as words per second using adjusted_running_time? 
# adjusted_running_time maybe time on page minus scrolling? maybe age? We'll inspect correlations with num_words.

for col in ['running_time','adjusted_running_time','age']:
    if col in df.columns:
        corr = df['num_words'].corr(df[col])
        print('corr num_words with', col, corr)

# Try compute reading speed
for col in ['running_time','adjusted_running_time','age']:
    if col in df.columns:
        speed = df['num_words'] / (df[col] / 1000.0)
        print(col, 'speed summary', speed.describe().to_string())
        break
