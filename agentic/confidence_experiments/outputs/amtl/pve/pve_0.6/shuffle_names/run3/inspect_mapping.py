import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 20)

df = pd.read_csv('amtl.csv')

# examine age by tooth class
print('age by tooth class (sockets column)')
print(df.groupby('sockets')['age'].describe())

print('\nnum_amtl by tooth class')
print(df.groupby('sockets')['num_amtl'].describe())

print('\npop by tooth class')
print(df.groupby('sockets')['pop'].describe())

print('\nmean age by genus (tooth_class column)')
print(df.groupby('tooth_class')['pop'].mean())

# check if age values correspond to standard counts per tooth class
print('\nunique age values by tooth class')
for tc, sub in df.groupby('sockets'):
    print(tc, sorted(sub['age'].unique())[:10], '... max', sub['age'].max())

# check if num_amtl is integer-like near integer
for col in ['genus','age','pop','num_amtl','stdev_age']:
    vals=df[col].values
    near_int = (abs(vals - np.round(vals)) < 1e-6).mean()
    print(col, 'near integer share', near_int)

