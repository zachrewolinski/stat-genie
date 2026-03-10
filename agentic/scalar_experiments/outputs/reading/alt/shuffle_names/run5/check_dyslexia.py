import pandas as pd
import numpy as np

_df = pd.read_csv('reading.csv')

# candidate dyslexia cols
candidates = ['device','dyslexia']

for col in candidates:
    # align non-missing with dyslexia_bin
    df = _df[[col,'dyslexia_bin']].dropna()
    # compute mismatch rate if dyslexia_bin is indicator of col>0
    indicator = (df[col] > 0).astype(float)
    mismatch = (indicator != df['dyslexia_bin']).mean()
    print(col, 'rows', len(df), 'mismatch_rate', mismatch)
    # contingency
    print(pd.crosstab(df[col], df['dyslexia_bin'], dropna=False))

# Compare to correct_rate as alternative binary
for col in candidates:
    df = _df[[col,'correct_rate']].dropna()
    indicator = (df[col] > 0).astype(float)
    mismatch = (indicator != df['correct_rate']).mean()
    print(col, 'vs correct_rate mismatch', mismatch)
    print(pd.crosstab(df[col], df['correct_rate'], dropna=False))

