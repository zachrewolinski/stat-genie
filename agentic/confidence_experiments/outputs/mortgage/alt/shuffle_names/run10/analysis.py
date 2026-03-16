import pandas as pd
import json

path = 'mortgage.csv'

df = pd.read_csv(path)
print(df.head())
print('columns', df.columns.tolist())
print('shape', df.shape)

# basic summary for each column
summary = []
for col in df.columns:
    s = df[col]
    summary.append({
        'col': col,
        'dtype': str(s.dtype),
        'n_unique': s.nunique(dropna=False),
        'min': s.min(),
        'max': s.max(),
        'mean': s.mean(),
        'std': s.std(),
        'value_counts': s.value_counts(dropna=False).head(5).to_dict()
    })

print('\nSummary:')
for item in summary:
    print(item)

