import pandas as pd
import numpy as np
from scipy.special import expit

df = pd.read_csv('amtl.csv')
p = expit(df['num_amtl'])
count_est = p * df['sockets']
# check how close to integer
frac = np.abs(count_est - np.round(count_est))
print('mean abs frac', frac.mean())
print('median abs frac', np.median(frac))
print('pct <0.1', (frac<0.1).mean())
print('count_est min/max', count_est.min(), count_est.max())
