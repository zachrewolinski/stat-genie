import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print('num_amtl mean/std', df['num_amtl'].mean(), df['num_amtl'].std())
print('age mean/std', df['age'].mean(), df['age'].std())
print('stdev_age mean/std', df['stdev_age'].mean(), df['stdev_age'].std())
print('prob_male unique', sorted(df['prob_male'].unique()))
