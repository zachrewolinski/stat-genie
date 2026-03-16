import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
# correlations
for col in ['genus','age','pop','num_amtl','stdev_age']:
    if col!='genus':
        print('corr genus vs', col, df['genus'].corr(df[col]))
# check if genus ~ linear to num_amtl/age
print('corr genus vs num_amtl/age', df['genus'].corr(df['num_amtl']/df['age']))
print('corr genus vs pop', df['genus'].corr(df['pop']))
print('corr num_amtl vs pop', df['num_amtl'].corr(df['pop']))
# check if num_amtl approx pop? (maybe age)
print('corr num_amtl vs pop', df['num_amtl'].corr(df['pop']))
