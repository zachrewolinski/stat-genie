import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')
vals = df['education'] / 1000.0
centers = np.array([0,1,2,3,7,12])
# distance to nearest center
nearest = centers[np.abs(vals.values[:,None]-centers).argmin(axis=1)]
dist = np.abs(vals - nearest)
print('mean dist', dist.mean())
print('median dist', dist.median())
print('within 0.1', (dist<=0.1).mean())
print('within 0.25', (dist<=0.25).mean())
print('within 0.5', (dist<=0.5).mean())
# show counts by nearest center
import collections
cnt = collections.Counter(nearest)
print(cnt)
