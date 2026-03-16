import pandas as pd
import numpy as np

_df = pd.read_csv('amtl.csv')
print('shape', _df.shape)
print(_df.head())
print(_df.dtypes)
print(_df[['num_amtl','sockets','age','prob_male']].head())
print(_df['num_amtl'].describe())
print(_df['sockets'].describe())
print(_df['genus'].value_counts())
print(_df['tooth_class'].value_counts())
