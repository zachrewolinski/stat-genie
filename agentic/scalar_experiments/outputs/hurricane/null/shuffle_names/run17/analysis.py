import pandas as pd
import numpy as np

path = 'hurricane.csv'

df = pd.read_csv(path)
print(df.head())
print('\nDtypes:')
print(df.dtypes)

print('\nDescribe (numeric):')
print(df.describe(include=[np.number]).T)

print('\nDescribe (object):')
print(df.describe(include=['object']).T)

print('\nMissing counts:')
print(df.isna().sum())
