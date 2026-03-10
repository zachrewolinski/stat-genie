import pandas as pd
import numpy as np


df = pd.read_csv('soccer.csv')

numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

summary = []
for c in numeric_cols:
    s = df[c].dropna()
    if len(s)==0:
        continue
    is_int = np.allclose(s % 1, 0)
    summary.append({
        'col': c,
        'min': s.min(),
        'max': s.max(),
        'mean': s.mean(),
        'std': s.std(),
        'zeros': (s==0).mean(),
        'nunique': s.nunique(),
        'is_int': is_int,
    })

summary_df = pd.DataFrame(summary).sort_values('mean')
print(summary_df.to_string(index=False))

# Show value counts for low-mean integer cols
cand = summary_df[(summary_df['is_int']) & (summary_df['mean']<1.0)].sort_values('mean')
print('\nlow-mean integer candidates')
print(cand[['col','min','max','mean','zeros','nunique']].to_string(index=False))

for c in cand['col']:
    print('\n', c)
    print(df[c].value_counts().sort_index().head(15))
