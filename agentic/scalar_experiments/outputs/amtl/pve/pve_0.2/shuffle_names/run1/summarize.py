import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
# individual-level grouping by specimen ID
by_spec = df.groupby('prob_male').first()
print('num specimens', by_spec.shape[0])
print('num_amtl summary', by_spec['num_amtl'].describe())
print('pop summary', by_spec['pop'].describe())
print('age summary (per row)', df['age'].describe())
print('genus summary', df['genus'].describe())
# check if num_amtl equals pop maybe? correlation
print('corr num_amtl vs pop', by_spec['num_amtl'].corr(by_spec['pop']))
# check if num_amtl is integer-ish
print('num_amtl unique fractional count', ((by_spec['num_amtl'] % 1) != 0).sum())
print('pop unique fractional count', ((by_spec['pop'] % 1) != 0).sum())
