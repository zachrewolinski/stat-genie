import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

for col in ['genus','pop','num_amtl']:
    frac = np.mean(df[col] <= df['age'])
    print(col, 'fraction <= age', frac)
    print('min difference', (df[col]-df['age']).min(), 'max difference', (df[col]-df['age']).max())

# Check for count-like by rounding and compare to age
for col in ['genus','pop','num_amtl']:
    rounded = np.round(df[col])
    frac = np.mean(rounded <= df['age'])
    print(col, 'rounded fraction <= age', frac)

