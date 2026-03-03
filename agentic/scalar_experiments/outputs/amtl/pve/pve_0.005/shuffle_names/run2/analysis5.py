import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

for col in ['age','pop','num_amtl','genus']:
    vals = df[col]
    # check how close to integer
    frac = np.abs(vals - np.round(vals))
    print(col, 'fraction near integer (<=1e-6):', (frac<=1e-6).mean())
    print('min frac', frac.min(), 'max frac', frac.max())
