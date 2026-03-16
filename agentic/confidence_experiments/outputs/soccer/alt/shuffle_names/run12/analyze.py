import pandas as pd
import numpy as np

path = 'soccer.csv'

df = pd.read_csv(path)
print(df.head())
print(df.columns)
print(df.dtypes)
print(df.shape)

# basic stats for potential skin tone columns (rater1, nExp)
for col in ['rater1','nExp']:
    if col in df.columns:
        print(col, df[col].describe())

# basic stats for redCards, yellowCards, yellowReds maybe
for col in ['yellowCards','yellowReds','redCards']:
    if col in df.columns:
        print(col, df[col].describe())
