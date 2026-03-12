import pandas as pd
import numpy as np

df = pd.read_csv('crofoot.csv')

print('columns', df.columns.tolist())
for col in df.columns:
    vals = df[col]
    print('\n', col)
    print(' min', vals.min(), 'max', vals.max(), 'unique', vals.nunique())
    if vals.nunique() <= 10:
        print(' value_counts', vals.value_counts().sort_index().to_dict())

binary = [c for c in df.columns if set(df[c].unique()) <= {0,1}]
print('\nBinary columns:', binary)

print('\nCorrelations with m_focal:')
print(df.corr(numeric_only=True)['m_focal'].sort_values(ascending=False))
