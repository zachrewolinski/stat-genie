import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print(df[['genus','age','pop','num_amtl','stdev_age']].corr())

# check unique values of num_amtl near integers? show quantiles
print('num_amtl quantiles', df['num_amtl'].quantile([0,0.1,0.25,0.5,0.75,0.9,1]))
print('pop quantiles', df['pop'].quantile([0,0.1,0.25,0.5,0.75,0.9,1]))

# check if num_amtl values maybe sample from 0-1? not

# show stdev_age value counts
print('stdev_age counts')
print(df['stdev_age'].value_counts().sort_index())

# check age value counts
print('age counts')
print(df['age'].value_counts().sort_index())

