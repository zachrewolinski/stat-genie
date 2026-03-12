import pandas as pd
import numpy as np


df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))
print('unique tooth_class', df['tooth_class'].unique())
print('unique sockets', df['sockets'].unique())
print('stdev_age unique', df['stdev_age'].unique())
print('prob_male unique sample', df['prob_male'].head())
