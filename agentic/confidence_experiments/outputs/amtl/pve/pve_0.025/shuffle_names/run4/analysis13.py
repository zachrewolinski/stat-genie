import pandas as pd
import numpy as np


df=pd.read_csv('amtl.csv')
# compute per specimen total sockets
sum_sockets = df.groupby('prob_male')['age'].sum()
num_amtl = df.groupby('prob_male')['num_amtl'].first()
prop = num_amtl / sum_sockets
print('prop range', prop.min(), prop.max())
print('prop >1 count', (prop>1).sum())
