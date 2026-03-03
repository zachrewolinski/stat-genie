import pandas as pd
import json

# load data
amtl = pd.read_csv('amtl.csv')
print(amtl.head())
print(amtl.dtypes)
print(amtl.shape)
print(amtl.columns)
print(amtl['genus'].unique()[:10])
print(amtl['tooth_class'].unique())
print(amtl['sockets'].unique())

# check missing
print(amtl.isna().sum())
