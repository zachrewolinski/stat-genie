import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')
num_cols = df.select_dtypes(include=[np.number]).columns

# consider integer-like columns
cand = []
for c in num_cols:
    s = df[c]
    if np.all(np.isclose(s.dropna() % 1, 0)):
        cand.append(c)

print('Integer-like numeric columns:', cand)

summary = []
for c in cand:
    s = df[c]
    summary.append((c, int(s.min()), int(s.max()), float(s.mean()), int((s==0).sum()), int(s.nunique())))

summary_sorted = sorted(summary, key=lambda x: (x[2], x[0]))
print('\nSummary (min, max, mean, zeros, nunique):')
for row in summary_sorted:
    print(row)

# show value counts for small max columns
print('\nValue counts for columns with max <= 10:')
for c, mn, mx, mean, zeros, nun in summary_sorted:
    if mx <= 10:
        counts = df[c].value_counts().sort_index().to_dict()
        print(c, counts)
