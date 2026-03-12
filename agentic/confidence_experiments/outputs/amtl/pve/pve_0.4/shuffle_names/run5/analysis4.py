import pandas as pd
import numpy as np

_df = pd.read_csv('amtl.csv')

# number of unique values per specimen for some columns
for col in ['num_amtl','pop','stdev_age']:
    uniques = _df.groupby('prob_male')[col].nunique()
    print(col, 'unique per specimen min/max', uniques.min(), uniques.max())

# for genus and age
for col in ['genus','age']:
    uniques = _df.groupby('prob_male')[col].nunique()
    print(col, 'unique per specimen min/max', uniques.min(), uniques.max())

# check average num_amtl by age categories to see relation
print(_df.groupby('age')['num_amtl'].mean().head())

