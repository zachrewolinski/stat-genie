import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

# candidate numerator columns
for col in ['genus','pop','num_amtl']:
    le = (df[col] <= df['age']).mean()
    print(col, 'fraction <= age:', le)
    print(col, 'min/max', df[col].min(), df[col].max())

# check if any column is within [0, age]
for col in ['genus','pop','num_amtl']:
    within = ((df[col] >= 0) & (df[col] <= df['age'])).mean()
    print(col, 'fraction within 0-age:', within)

# check if num_amtl maybe age stdev? compare to pop
print('corr pop-num_amtl', df['pop'].corr(df['num_amtl']))
print('corr pop-genus', df['pop'].corr(df['genus']))
print('corr num_amtl-genus', df['num_amtl'].corr(df['genus']))
