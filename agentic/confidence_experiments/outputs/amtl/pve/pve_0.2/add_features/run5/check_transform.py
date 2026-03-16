import pandas as pd
import numpy as np
from scipy.special import expit

df = pd.read_csv('amtl.csv')
# estimate counts if num_amtl is logit proportion
p = expit(df['num_amtl'])
counts = p * df['sockets']
# measure closeness to integer
frac = np.abs(counts - np.round(counts))
print('mean frac', frac.mean(), 'median', np.median(frac))
print('pct close <0.1', (frac<0.1).mean())
print('counts min/max', counts.min(), counts.max())
