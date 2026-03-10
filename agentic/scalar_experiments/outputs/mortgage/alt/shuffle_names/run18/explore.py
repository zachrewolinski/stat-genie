import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 50)

_df = pd.read_csv('mortgage.csv')

binary_cols = [c for c in _df.columns if _df[c].dropna().isin([0,1]).all()]
print('Binary cols:', binary_cols)
print('\nBinary means:')
for c in binary_cols:
    print(c, _df[c].mean())

print('\nDescribe numeric:')
print(_df.describe().T[['mean','std','min','max']])

# Check correlation between accept/deny and other binaries
if 'accept' in _df.columns and 'deny' in _df.columns:
    print('\naccept/deny correlation', _df['accept'].corr(_df['deny']))

# Maybe there is variable for approval? check if any binary column close to 0.8 acceptance?
# We'll compute approval rate by binary columns to see if accept or deny align

# Check if any binary column is mostly zero or one

# Look at missing values
print('\nMissing values:')
print(_df.isna().sum())

# Check whether accept and deny are complements? if not maybe accept is something else
if 'accept' in _df.columns and 'deny' in _df.columns:
    comp = (_df['accept'] == (1 - _df['deny'])).mean()
    print('Share accept == 1 - deny', comp)

# Check values of "Unnamed: 0"
print('\nUnnamed: 0 range', _df['Unnamed: 0'].min(), _df['Unnamed: 0'].max())

# Inspect distribution of mortgage_credit/housing_expense_ratio
for c in ['mortgage_credit','housing_expense_ratio']:
    if c in _df.columns:
        print('\n', c, 'range', _df[c].min(), _df[c].max())

# Find columns with values similar to acceptance rate about 0.8 or 0.2
# Not sure.
