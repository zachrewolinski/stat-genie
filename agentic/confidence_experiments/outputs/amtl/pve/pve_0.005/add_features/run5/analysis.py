import pandas as pd
import numpy as np


df = pd.read_csv('amtl.csv')
print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head())
print('\nsummary num_amtl:')
print(df['num_amtl'].describe())
print('num_amtl non-integer count:', (~np.isclose(df['num_amtl'], np.round(df['num_amtl']))).sum())
print('num_amtl min/max:', df['num_amtl'].min(), df['num_amtl'].max())
print('sockets describe', df['sockets'].describe())
print('sockets non-integer count:', (~np.isclose(df['sockets'], np.round(df['sockets']))).sum())
print('unique genus', df['genus'].unique())
print('tooth_class unique', df['tooth_class'].unique())
print('prob_male unique', sorted(df['prob_male'].unique())[:10])
print('age describe', df['age'].describe())

# check for missing
print('missing per column', df.isna().sum())
