import pandas as pd
import numpy as np


df = pd.read_csv('amtl.csv')
print('shape', df.shape)
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))
print('num_amtl stats', df['num_amtl'].min(), df['num_amtl'].max())
print('sockets stats', df['sockets'].min(), df['sockets'].max())
print('unique genus', df['genus'].unique())
print('tooth_class unique', df['tooth_class'].unique())
