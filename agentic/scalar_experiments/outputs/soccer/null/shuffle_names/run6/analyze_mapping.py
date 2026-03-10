import pandas as pd
import numpy as np

path = 'soccer.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)

# For each column, show basic stats for numeric columns
summary = []
for col in df.columns:
    s = df[col]
    # attempt to coerce to numeric
    s_num = pd.to_numeric(s, errors='coerce')
    num_non_na = s_num.notna().sum()
    if num_non_na > 0:
        summary.append((col, num_non_na, s_num.min(), s_num.max(), s_num.nunique(), s.nunique()))

print('numeric summary: col, num_non_na, min, max, nunique_num, nunique_raw')
for row in summary:
    print(row)

# show columns with date-like pattern dd.mm.yyyy
import re
for col in df.columns:
    if df[col].dtype == object:
        sample = df[col].dropna().astype(str).head(50)
        if sample.str.match(r"\d{2}\.\d{2}\.\d{4}").any():
            print('date-like column', col, sample[sample.str.match(r"\d{2}\.\d{2}\.\d{4}")].head().tolist())

# show columns with values between 0 and 1
for col in df.columns:
    s_num = pd.to_numeric(df[col], errors='coerce')
    if s_num.notna().sum() > 0:
        if s_num.min() >= 0 and s_num.max() <= 1:
            print('0-1 column', col, 'unique', sorted(s_num.dropna().unique())[:10], 'nunique', s_num.nunique())
