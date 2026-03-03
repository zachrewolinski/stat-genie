import pandas as pd
import numpy as np

# Load data

df = pd.read_csv('amtl.csv')

print('Head:')
print(df.head())

print('\nDescribe:')
print(df.describe(include='all'))

print('\nMissing values:')
print(df.isna().sum())

# summary for counts
print('\nfeature3 min/max', df['feature3'].min(), df['feature3'].max())
print('feature4 min/max', df['feature4'].min(), df['feature4'].max())

# proportion
prop = df['feature3'] / df['feature4']
print('prop min/max', prop.min(), prop.max())

# check invalid counts (negative or > feature4)
invalid = (df['feature3'] < 0) | (df['feature3'] > df['feature4'])
print('invalid count', invalid.sum())

# check genus counts
print('\nGenus counts:')
print(df['feature8'].value_counts())
