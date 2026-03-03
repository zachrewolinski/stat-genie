import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

# check integer-ish
for col in ['genus','age','pop','num_amtl','stdev_age']:
    vals = df[col].dropna()
    # difference from nearest int
    diff = (vals - vals.round()).abs().max()
    print(col, 'max abs diff from int', diff)

# summarize by tooth_class (genus)
print('\nMeans by tooth_class (genus categories):')
print(df.groupby('tooth_class')[['genus','age','pop','num_amtl','stdev_age']].mean())

# check min/max by tooth_class
print('\nMin/max num_amtl by tooth_class:')
print(df.groupby('tooth_class')['num_amtl'].agg(['min','max','mean','std']))

# check correlation between numeric columns
print('\nCorrelation:')
print(df[['genus','age','pop','num_amtl','stdev_age']].corr())

# see distribution of stdev_age (0-1?)
print('\nStdev_age value counts:')
print(df['stdev_age'].value_counts().sort_index())

# unique age values
print('\nAge value counts (first 20):')
print(df['age'].value_counts().sort_index().head(20))

