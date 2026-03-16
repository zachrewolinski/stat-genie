import pandas as pd
import numpy as np

path = 'soccer.csv'
df = pd.read_csv(path)

print('shape', df.shape)
print(df.head(3))

# summary for each column
summary = []
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        summary.append({
            'col': col,
            'dtype': 'num',
            'min': np.nanmin(s.values),
            'max': np.nanmax(s.values),
            'mean': np.nanmean(s.values),
            'std': np.nanstd(s.values),
            'n_unique': s.nunique(dropna=True),
            'n_missing': s.isna().sum(),
        })
    else:
        summary.append({
            'col': col,
            'dtype': 'cat',
            'n_unique': s.nunique(dropna=True),
            'n_missing': s.isna().sum(),
            'sample_vals': s.dropna().unique()[:5].tolist()
        })

# print numeric summaries sorted by min/max
print('\nNUMERIC SUMMARY')
num_df = pd.DataFrame([x for x in summary if x['dtype']=='num']).sort_values('max')
print(num_df.to_string(index=False))

print('\nCATEGORICAL SUMMARY')
cat_df = pd.DataFrame([x for x in summary if x['dtype']=='cat'])
print(cat_df.to_string(index=False))

# show unique values for columns with small unique counts
print('\nSMALL UNIQUE NUMERIC COLS (<=10 unique)')
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) <= 10:
        print(col, sorted(s.dropna().unique().tolist()))

