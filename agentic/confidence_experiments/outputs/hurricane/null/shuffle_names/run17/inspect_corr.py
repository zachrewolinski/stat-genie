import pandas as pd
import numpy as np

df=pd.read_csv('hurricane.csv')

print('corr wind(year?) with masfem:', df['wind'].corr(df['masfem']))
print('corr wind with masfem (if masfem is elapsed yrs), should be high')

print('corr wind with masfem if masfem is years since 1950:')

# calculate elapsed years from wind (year) to see if matches masfem
elapsed = df['wind'] - df['wind'].min()
print('min year', df['wind'].min(), 'max', df['wind'].max())
print('elapsed stats', elapsed.describe())
print('masfem stats', df['masfem'].describe())

# correlation between elapsed and masfem
print('corr elapsed vs masfem', elapsed.corr(df['masfem']))

# check if masfem equals elapsed within rounding
print('max abs diff between elapsed and masfem', (elapsed - df['masfem']).abs().max())

