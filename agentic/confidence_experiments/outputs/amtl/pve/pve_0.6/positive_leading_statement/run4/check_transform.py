import pandas as pd
import numpy as np
from scipy.special import expit

df = pd.read_csv('amtl.csv')
# check if expit(num_amtl)*sockets near integer
p = expit(df['num_amtl'].to_numpy())
count = p * df['sockets'].to_numpy()
# distance to nearest integer
dist = np.abs(count - np.round(count))
print('mean dist', dist.mean(), 'median', np.median(dist))
print('pct within 0.05', np.mean(dist < 0.05))
print('pct within 0.1', np.mean(dist < 0.1))
# check if maybe num_amtl is proportion itself? then p = num_amtl, count = num_amtl*sockets
p2 = df['num_amtl'].to_numpy()
count2 = p2 * df['sockets'].to_numpy()
# for proportions outside [0,1], ignore
mask = (p2>=0) & (p2<=1)
print('prop within [0,1]', mask.mean())
if mask.any():
    dist2 = np.abs(count2[mask] - np.round(count2[mask]))
    print('proportion model mean dist', dist2.mean(), 'pct within 0.05', np.mean(dist2<0.05))

# maybe num_amtl is log(count+0.5)?? try exponentiate and see closeness
count3 = np.exp(df['num_amtl'].to_numpy())
print('exp num_amtl range', count3.min(), count3.max())
