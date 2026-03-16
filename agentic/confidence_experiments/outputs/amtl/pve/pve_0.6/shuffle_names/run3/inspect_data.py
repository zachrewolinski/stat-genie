import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 20)


df = pd.read_csv('amtl.csv')
print('shape', df.shape)
print(df.head())
print('\nsummary numeric')
print(df.describe())
print('\nvalue counts for tooth_class')
print(df['tooth_class'].value_counts())
print('\nvalue counts for sockets')
print(df['sockets'].value_counts())

# correlations among numeric
print('\ncorrelation')
print(df.select_dtypes('number').corr())

# Check integer-like
for col in ['genus','age','pop','num_amtl','stdev_age']:
    series=df[col]
    if np.issubdtype(series.dtype, np.number):
        print(col, 'nunique', series.nunique(), 'rounded unique', series.round(3).nunique())
        print(col, 'min', series.min(), 'max', series.max())

