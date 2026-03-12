import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

# check comparisons
print('num_amtl <= age', (df['num_amtl'] <= df['age']).mean())
print('genus <= age', (df['genus'] <= df['age']).mean())
print('num_amtl <= pop', (df['num_amtl'] <= df['pop']).mean())

# check if age could be sockets count and num_amtl missing count by seeing fraction of rows where num_amtl is integer
for col in ['genus','num_amtl']:
    frac_int = (np.abs(df[col] - np.round(df[col])) < 1e-6).mean()
    print(col, 'frac int', frac_int)

# check if any numeric column is between 0 and 1 exclusively
for col in ['genus','pop','num_amtl','stdev_age']:
    print(col, 'range', df[col].min(), df[col].max())

# check if stdev_age could be probability male by unique values and relation to tooth_class
print('stdev_age value counts', df['stdev_age'].value_counts())

