import pandas as pd
import numpy as np

pd.set_option('display.width', 120)

# Load data

df = pd.read_csv('amtl.csv')
print('Head:')
print(df.head())
print('\nDescribe numeric:')
print(df.describe())
print('\nUnique counts:')
print(df.nunique())

# Check ranges for candidate columns
for col in ['genus','age','pop','num_amtl','stdev_age']:
    print(f"\n{col} min/max/mean:", df[col].min(), df[col].max(), df[col].mean())

# Cross-tab counts by tooth_class (genus) and sockets (tooth class)
print('\nCounts by tooth_class and sockets:')
print(pd.crosstab(df['tooth_class'], df['sockets']))

# Inspect stdev_age values
print('\nstdev_age value counts:')
print(df['stdev_age'].value_counts().sort_index())

# Inspect age value counts
print('\nage value counts:')
print(df['age'].value_counts().sort_index())

# Inspect pop distribution
print('\npop quantiles:')
print(df['pop'].quantile([0,0.1,0.25,0.5,0.75,0.9,1.0]))

# inspect num_amtl distribution
print('\nnum_amtl quantiles:')
print(df['num_amtl'].quantile([0,0.1,0.25,0.5,0.75,0.9,1.0]))

