import pandas as pd
import numpy as np


df = pd.read_csv('amtl.csv')
for col in ['genus','age','pop','num_amtl','stdev_age']:
    vals = df[col].values
    diff = np.abs(vals - np.round(vals))
    print(col, 'mean abs diff', diff.mean(), 'max abs diff', diff.max())

