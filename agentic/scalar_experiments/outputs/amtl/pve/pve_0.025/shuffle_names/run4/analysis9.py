import pandas as pd
import numpy as np


df=pd.read_csv('amtl.csv')
num_cols=['genus','age','pop','num_amtl','stdev_age']

for num in num_cols:
    for denom in num_cols:
        if num==denom:
            continue
        ratio = df[num]/df[denom]
        frac = np.mean((ratio >= 0) & (ratio <= 1))
        print(f'{num}/{denom}: frac in [0,1]={frac:.3f}')
