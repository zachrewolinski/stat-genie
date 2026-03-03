import pandas as pd
amtl = pd.read_csv('amtl.csv')
print(amtl.describe(include='all'))

for col in ['genus','age','pop','num_amtl','stdev_age']:
    print('\n',col)
    print(amtl[col].describe())
    print('unique count',amtl[col].nunique())
    print('min max',amtl[col].min(),amtl[col].max())

# check integer-like columns
for col in ['genus','age','pop','num_amtl','stdev_age']:
    unique_frac = (amtl[col] % 1 == 0).mean()
    print(col, 'fraction integer', unique_frac)

