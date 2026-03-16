import pandas as pd
import numpy as np

df = pd.read_csv('mortgage.csv')
summary = []
for col in df.columns:
    series = df[col]
    nunique = series.nunique(dropna=True)
    summary.append((col, nunique, series.min(), series.max(), series.mean()))

summary_sorted = sorted(summary, key=lambda x: x[1])
print('Column summary (sorted by nunique):')
for col, nunique, minv, maxv, meanv in summary_sorted:
    print(f"{col:20} nunique={nunique:4} min={minv:8} max={maxv:8} mean={meanv:8.4f}")

# Identify binary columns
print('\nBinary columns:')
for col, nunique, minv, maxv, meanv in summary_sorted:
    if nunique == 2:
        vc = df[col].value_counts().to_dict()
        print(col, vc, 'mean', meanv)

# Identify near-binary or small categorical (<=6 unique)
print('\nSmall categorical columns (<=6 unique):')
for col, nunique, minv, maxv, meanv in summary_sorted:
    if nunique <= 6:
        vc = df[col].value_counts().head(10).to_dict()
        print(col, 'nunique', nunique, 'values', vc)
