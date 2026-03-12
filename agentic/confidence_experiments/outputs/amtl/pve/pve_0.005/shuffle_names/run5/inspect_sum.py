import pandas as pd

df = pd.read_csv('amtl.csv')
# sum genus across rows per specimen
sum_genus = df.groupby('prob_male')['genus'].sum()
num_amtl = df.groupby('prob_male')['num_amtl'].first()
# correlation and difference stats
import numpy as np
print('corr', sum_genus.corr(num_amtl))
print('mean diff', (sum_genus - num_amtl).mean())
print('std diff', (sum_genus - num_amtl).std())
print('min diff', (sum_genus - num_amtl).min(), 'max diff', (sum_genus - num_amtl).max())
