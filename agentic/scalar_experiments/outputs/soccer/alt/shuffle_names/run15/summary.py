import pandas as pd
import numpy as np

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 50)


df = pd.read_csv('soccer.csv')

num_cols = df.select_dtypes(include=[np.number]).columns

summary = []
for c in num_cols:
    s = df[c]
    summary.append({
        'col': c,
        'min': s.min(),
        'max': s.max(),
        'mean': s.mean(),
        'std': s.std(),
        'n_unique': s.nunique(dropna=True),
        'pct_zero': float((s==0).mean()),
        'pct_na': float(s.isna().mean()),
    })

summary_df = pd.DataFrame(summary).sort_values('col')
print(summary_df[['col','min','max','mean','std','n_unique','pct_zero','pct_na']])

print('\nSmall-cardinality numeric columns:')
small = summary_df[summary_df['n_unique']<=10]
print(small[['col','min','max','n_unique']])

print('\nSorted by max:')
print(summary_df.sort_values('max')[['col','min','max','n_unique']])

