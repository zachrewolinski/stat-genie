import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
prop = 1/(1+np.exp(-df['num_amtl']))
prod = prop * df['sockets']
close_int = (np.abs(prod - np.round(prod)) < 0.05)
print('close to integer proportion after logit', close_int.mean())
print('min max prop', prop.min(), prop.max())
print('min max prod', prod.min(), prod.max())
