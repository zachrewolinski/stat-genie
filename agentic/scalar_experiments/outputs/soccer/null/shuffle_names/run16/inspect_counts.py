import pandas as pd
import numpy as np

path='soccer.csv'
df=pd.read_csv(path)

int_cols=[c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
print('int cols', int_cols)
for c in int_cols:
    vc=df[c].value_counts().sort_index()
    print('\n',c,'min',df[c].min(),'max',df[c].max(),'mean',df[c].mean())
    print(vc.head(10))
    print('...')
    print(vc.tail(5))
