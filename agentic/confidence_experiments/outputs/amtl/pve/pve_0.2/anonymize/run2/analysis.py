import pandas as pd
import json

# Load data
amtl = pd.read_csv('amtl.csv')

# Basic info
print('shape', amtl.shape)
print(amtl.head())
print('dtypes', amtl.dtypes)

# summarize key columns
cols = amtl.columns.tolist()
print('columns', cols)

# inspect feature8 (genus)
print('feature8 unique', amtl['feature8'].unique())

# summary stats for feature3 and feature4
print('feature3 summary', amtl['feature3'].describe())
print('feature4 summary', amtl['feature4'].describe())

# check range of missing values
print('feature3 min max', amtl['feature3'].min(), amtl['feature3'].max())
print('feature4 min max', amtl['feature4'].min(), amtl['feature4'].max())

# check if feature3 integer? 
print('feature3 integer?', (amtl['feature3'] % 1).abs().max())
print('feature4 integer?', (amtl['feature4'] % 1).abs().max())

# check if feature3 maybe proportion? 
print('feature3 proportion range', (amtl['feature3'] / amtl['feature4']).describe())

# check missing values
print('missing counts', amtl.isna().sum())

