import pandas as pd
import numpy as np

_df = pd.read_csv('amtl.csv')
print(_df.head())
print(_df.dtypes)
print(_df.describe(include='all'))

# check integer-like for numeric columns
for col in _df.columns:
    if pd.api.types.is_numeric_dtype(_df[col]):
        vals = _df[col].dropna()
        frac = np.mean(np.isclose(vals, np.round(vals)))
        print(col, 'min', vals.min(), 'max', vals.max(), 'mean', vals.mean(), 'frac_int', frac)

# unique values for stdev_age and age etc
for col in ['age','stdev_age']:
    if col in _df.columns:
        print(col, sorted(_df[col].unique())[:20])

# check counts by sockets category
print(_df['sockets'].value_counts())
print(_df['tooth_class'].value_counts())
