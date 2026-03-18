import pandas as pd
import numpy as np


df = pd.read_csv('affairs.csv')

for col in ['age', 'education']:
    for decimals in [0, 1, 2]:
        rounded = df[col].round(decimals)
        print(col, 'rounded', decimals, 'unique', rounded.nunique(), 'min', rounded.min(), 'max', rounded.max())
    rounded0 = df[col].round(0)
    print(col, 'top rounded0')
    print(rounded0.value_counts().head(10))
    print('---')
