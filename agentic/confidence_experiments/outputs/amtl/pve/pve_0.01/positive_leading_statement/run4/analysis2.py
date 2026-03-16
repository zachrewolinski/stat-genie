import pandas as pd
import numpy as np

amtl = pd.read_csv('amtl.csv')
# test if num_amtl is logit of proportion
from scipy.special import expit
amtl['p']=expit(amtl['num_amtl'])
# approximate counts
amtl['count_est']=amtl['p']*amtl['sockets']
print(amtl[['num_amtl','sockets','p','count_est']].head())
print('count_est range', amtl['count_est'].min(), amtl['count_est'].max())
# check if count_est close to integer? compute mean abs difference to nearest int
nearest = amtl['count_est'].round()
print('mean abs diff to nearest int', (amtl['count_est']-nearest).abs().mean())
# check if num_amtl relates to standardization of count
# Try infer original count as (num_amtl*sd + mean) from groups? No.

