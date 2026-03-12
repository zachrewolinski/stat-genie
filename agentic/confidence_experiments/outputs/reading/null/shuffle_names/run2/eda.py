import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_csv('reading.csv')
print('shape', df.shape)
print('columns', df.columns.tolist())

for col in df.columns:
    s = df[col]
    nunique = s.nunique(dropna=False)
    print('\n', col)
    print('  dtype', s.dtype, 'nunique', nunique)
    if s.dtype == 'object':
        print('  sample', s.dropna().unique()[:10])
    else:
        print('  min', s.min(), 'max', s.max())
        print('  mean', s.mean())
        print('  sample', s.dropna().unique()[:10])
