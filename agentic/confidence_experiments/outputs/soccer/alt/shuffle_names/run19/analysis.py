import pandas as pd
import numpy as np

path = 'soccer.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())
print('\ncolumns')
print(df.columns.tolist())

# summary per column
summary = []
for col in df.columns:
    s = df[col]
    summary.append({
        'col': col,
        'dtype': s.dtype,
        'n_unique': s.nunique(dropna=True),
        'n_missing': s.isna().sum(),
        'min': s.min() if pd.api.types.is_numeric_dtype(s) else None,
        'max': s.max() if pd.api.types.is_numeric_dtype(s) else None,
        'sample': s.dropna().head(5).tolist(),
    })

summary_df = pd.DataFrame(summary)
print('\nsummary')
print(summary_df)

# show potential skin tone columns: numeric with limited unique values ~5
print('\npossible skin tone columns (n_unique<=7, numeric)')
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) <= 7:
        print(col, 'unique', sorted(s.dropna().unique())[:10])

# possible red card columns: count data, small integer, maybe 0..?
print('\npossible red card columns (integer-like, max <=10)')
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        vals = s.dropna()
        if len(vals) == 0:
            continue
        if (np.all(np.isclose(vals, np.round(vals)))) and vals.max() <= 10:
            print(col, 'min', vals.min(), 'max', vals.max(), 'unique', sorted(vals.unique())[:10])

# possible game counts (matches) big >10
print('\npossible games columns (max > 20)')
for col in df.columns:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        vals = s.dropna()
        if len(vals) == 0:
            continue
        if vals.max() > 20:
            print(col, 'min', vals.min(), 'max', vals.max(), 'unique sample', sorted(vals.unique())[:5])

