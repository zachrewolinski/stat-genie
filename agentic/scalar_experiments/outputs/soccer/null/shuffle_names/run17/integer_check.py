import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        s = df[col].dropna()
        # check if all values are close to integers
        is_int = np.all(np.isclose(s, np.round(s)))
        if is_int:
            print(col, 'integer', 'min', s.min(), 'max', s.max(), 'unique', s.nunique())
        else:
            # report non-integer columns
            print(col, 'non-integer', 'min', s.min(), 'max', s.max(), 'unique', s.nunique())
