import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# Load data

df = pd.read_csv('amtl.csv')

# Summary by genus (tooth_class column)
print('Genus counts:')
print(df['tooth_class'].value_counts())

# For numeric columns, compute mean by genus
num_cols = df.select_dtypes(exclude='object').columns
print('\nMeans by genus:')
print(df.groupby('tooth_class')[num_cols].mean())

# Correlations among numeric columns
print('\nCorrelation matrix:')
print(df[num_cols].corr())

# Check if any numeric columns are bounded by age count, etc.
# Compute ratios if possible
for col in ['genus','pop','num_amtl']:
    ratio = df[col] / df['age']
    print(f"\n{col}/age summary:")
    print(ratio.describe())

# Look at min/max by tooth class for numeric columns
print('\nMin/Max by genus for numeric columns:')
print(df.groupby('tooth_class')[num_cols].agg(['min','max']))

# Check if num_amtl is integerish by rounding
for col in num_cols:
    rounded = np.round(df[col])
    diff = np.abs(df[col] - rounded)
    print(col, 'mean abs diff to nearest int', diff.mean(), 'max diff', diff.max())

