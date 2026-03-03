import pandas as pd
import numpy as np

amtl = pd.read_csv('amtl.csv')
print('corr num_amtl vs sockets', amtl['num_amtl'].corr(amtl['sockets']))
print('corr num_amtl vs age', amtl['num_amtl'].corr(amtl['age']))
print('corr num_amtl vs prob_male', amtl['num_amtl'].corr(amtl['prob_male']))

# try to see if num_amtl is standardized within tooth_class
for cls in amtl['tooth_class'].unique():
    sub = amtl[amtl['tooth_class']==cls]
    print(cls, 'mean', sub['num_amtl'].mean(), 'std', sub['num_amtl'].std())

# maybe num_amtl = (count - sockets*mean_rate) / sd? Hard.

