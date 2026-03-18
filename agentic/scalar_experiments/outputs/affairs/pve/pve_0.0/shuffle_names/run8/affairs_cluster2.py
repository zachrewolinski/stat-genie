import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')
vals = df['education'] / 1000.0
centers = np.array([0,1,2,3,7,12])
for c in centers:
    within = (np.abs(vals - c) <= 0.5).mean()
    print(c, within)
# show max
print('max', vals.max(), 'min', vals.min())
