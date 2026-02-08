import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')
print(df.head())
print(df.dtypes)
for col in df.columns:
    ser = df[col]
    print('\n', col)
    print('nunique', ser.nunique(dropna=False))
    print('min', ser.min(), 'max', ser.max())
    # show value counts for small unique
    if ser.nunique() <= 10:
        print(ser.value_counts(dropna=False).sort_index())
    else:
        print('sample', ser.sample(5, random_state=0).tolist())
