import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')
print('shape', df.shape)
print('columns', df.columns.tolist())

num_cols = df.select_dtypes(include=[np.number]).columns
summary = []
for c in num_cols:
    s = df[c]
    summary.append((c, float(s.min()), float(s.max()), float(s.mean()), int(s.nunique())))
summary_sorted = sorted(summary, key=lambda x: (x[2]-x[1], x[1], x[0]))
print('\nNumeric columns summary (min, max, mean, nunique):')
for row in summary_sorted:
    print(row)

print('\nCandidate discrete columns with <=6 unique values:')
for c in num_cols:
    nun = df[c].nunique()
    if nun <= 6:
        vals = df[c].value_counts().sort_index()
        print(c, 'nunique', nun, 'values', vals.to_dict())

print('\nHead:')
print(df.head())
