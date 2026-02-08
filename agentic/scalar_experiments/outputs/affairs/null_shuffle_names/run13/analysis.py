import pandas as pd
import numpy as np
from math import sqrt

# Load data

df = pd.read_csv('affairs.csv')

# Based on metadata/values: 'age' encodes affairs frequency (0,1,2,3,7,12)
# and 'religiousness' encodes whether there are children (yes/no).

# Map children variable
children = df['religiousness'].map({'yes': 1, 'no': 0})
# Affairs frequency
affairs = df['age']

# Basic group stats
stats = df.groupby('religiousness')['age'].agg(['count','mean','std'])
print('Group stats (affairs frequency by children):')
print(stats)

# Any-affair indicator
any_affair = (affairs > 0).astype(int)
any_stats = df.assign(any_affair=any_affair).groupby('religiousness')['any_affair'].agg(['count','mean'])
print('\nAny-affair rate by children:')
print(any_stats)

# Difference in means (no children - children)
mean_no = stats.loc['no','mean']
mean_yes = stats.loc['yes','mean']
mean_diff = mean_no - mean_yes

# Cohen's d
std_no = stats.loc['no','std']
std_yes = stats.loc['yes','std']
n_no = stats.loc['no','count']
n_yes = stats.loc['yes','count']

pooled_var = ((n_no-1)*std_no**2 + (n_yes-1)*std_yes**2) / (n_no + n_yes - 2)
cohen_d = mean_diff / sqrt(pooled_var)

# Risk difference for any affair
rate_no = any_stats.loc['no','mean']
rate_yes = any_stats.loc['yes','mean']
rd = rate_no - rate_yes

print('\nMean diff (no - yes):', mean_diff)
print('Cohen d:', cohen_d)
print('Any-affair rate diff (no - yes):', rd)
