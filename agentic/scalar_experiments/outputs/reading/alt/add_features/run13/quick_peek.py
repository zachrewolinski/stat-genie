import pandas as pd
import numpy as np

path = 'reading.csv'

df = pd.read_csv(path)
print(df.head())
print(df.columns)
print(df.shape)
print(df['dyslexia'].unique()[:10])
print(df['dyslexia_bin'].value_counts(dropna=False))
print(df['reader_view'].value_counts(dropna=False))
print(df['speed'].describe())
