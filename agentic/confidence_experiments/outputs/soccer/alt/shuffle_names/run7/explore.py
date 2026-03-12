import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print('Numeric columns:', num_cols)

summary = []
for col in num_cols:
    s = df[col]
    summary.append({
        'col': col,
        'min': s.min(),
        'max': s.max(),
        'mean': s.mean(),
        'std': s.std(),
        'p25': s.quantile(0.25),
        'p50': s.quantile(0.5),
        'p75': s.quantile(0.75),
        'p95': s.quantile(0.95),
        'pct_zero': float((s==0).mean()),
        'pct_na': float(s.isna().mean()),
        'nunique': s.nunique(dropna=True),
    })

summary_df = pd.DataFrame(summary).sort_values(['mean'])
print('\nSummary sorted by mean:')
print(summary_df.to_string(index=False))

print('\nLow-unique numeric columns values:')
for col in num_cols:
    nunique = df[col].nunique(dropna=True)
    if nunique <= 10:
        vals = sorted(df[col].dropna().unique())
        print(col, nunique, vals)
